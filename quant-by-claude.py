"""
TinyNPU: Train FP32 -> Quantize (PTQ) -> Fine-Tune (QAT) -> Export
==================================================================
3-stage pipeline optimized for TinyNPU's INT4/8/16 hardware.

Usage:
  python quant-by-claude.py --train                         # Stage 1: FP32 training
  python quant-by-claude.py --quantize                      # Stage 2: PTQ calibration + eval
  python quant-by-claude.py --finetune                      # Stage 3: QAT fine-tuning
  python quant-by-claude.py --sensitivity                   # Sensitivity analysis
  python quant-by-claude.py --export --export-dir ./out     # Export for NPU
  python quant-by-claude.py --conv2-bits 4 4                # Per-layer bit config
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import json
import os
import argparse
import math
import copy

from software.compiler.tinynpu_quant import compute_fused_params

# ============================================================================
# 1. FP32 MODEL (Stage 1)
# ============================================================================

class TinyNetFP32(nn.Module):
    """Plain FP32 model for baseline training."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1, bias=True)
        self.fc = nn.Linear(16, 10, bias=True)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.mean(dim=[2, 3])  # Global average pooling
        x = self.fc(x)
        return x

# ============================================================================
# 2. FAKE QUANTIZATION (for QAT)
# ============================================================================

class SymmetricQuantizer(torch.autograd.Function):
    """STE-based fake quantizer."""
    @staticmethod
    def forward(ctx, x, scale, num_bits, is_weight):
        if is_weight:
            qmin = -(2 ** (num_bits - 1)) + 1
            qmax = 2 ** (num_bits - 1) - 1
        else:
            qmin = 0
            qmax = 2 ** num_bits - 1
        x_scaled = x / scale
        x_clamped = torch.clamp(x_scaled, qmin, qmax)
        x_quant = torch.round(x_clamped)
        x_dequant = x_quant * scale
        ctx.save_for_backward(x_scaled, torch.tensor(qmin), torch.tensor(qmax))
        return x_dequant

    @staticmethod
    def backward(ctx, grad_output):
        x_scaled, qmin, qmax = ctx.saved_tensors
        mask = (x_scaled >= qmin.float()) & (x_scaled <= qmax.float())
        return grad_output * mask.float(), None, None, None

def fake_quantize(x, scale, num_bits, is_weight=False):
    return SymmetricQuantizer.apply(x, scale, num_bits, is_weight)

# ============================================================================
# 3. QAT MODEL (Stage 3)
# ============================================================================

class QConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding=0, stride=1, w_bits=8, a_bits=8):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, stride=stride, bias=True)
        self.w_bits, self.a_bits = w_bits, a_bits
        self.w_scale = nn.Parameter(torch.tensor(0.05))
        self.a_scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, x):
        # Update weight scale from current weights
        with torch.no_grad():
            w_qmax = 2 ** (self.w_bits - 1) - 1
            self.w_scale.data.copy_((self.conv.weight.abs().max() / w_qmax).clamp(min=1e-8))
        x_q = fake_quantize(x, self.a_scale, self.a_bits, is_weight=False)
        w_q = fake_quantize(self.conv.weight, self.w_scale, self.w_bits, is_weight=True)
        return F.conv2d(x_q, w_q, self.conv.bias, self.conv.stride, self.conv.padding)

class QLinear(nn.Module):
    def __init__(self, in_features, out_features, w_bits=8, a_bits=8):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.w_bits, self.a_bits = w_bits, a_bits
        self.w_scale = nn.Parameter(torch.tensor(0.05))
        self.a_scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, x):
        with torch.no_grad():
            w_qmax = 2 ** (self.w_bits - 1) - 1
            self.w_scale.data.copy_((self.linear.weight.abs().max() / w_qmax).clamp(min=1e-8))
        x_q = fake_quantize(x, self.a_scale, self.a_bits, is_weight=False)
        w_q = fake_quantize(self.linear.weight, self.w_scale, self.w_bits, is_weight=True)
        return F.linear(x_q, w_q, self.linear.bias)

class TinyNetQAT(nn.Module):
    """Quantization-aware model, initialized from FP32 weights + calibrated scales."""
    def __init__(self, layer_configs=None):
        super().__init__()
        if layer_configs is None:
            layer_configs = {k: {'w_bits': 8, 'a_bits': 8} for k in ['conv1', 'conv2', 'conv3', 'fc']}
        c = layer_configs
        self.conv1 = QConv2d(1, 16, 3, padding=1, **c['conv1'])
        self.conv2 = QConv2d(16, 16, 3, padding=1, **c['conv2'])
        self.conv3 = QConv2d(16, 16, 3, padding=1, **c['conv3'])
        self.fc = QLinear(16, 10, **c['fc'])

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.mean(dim=[2, 3])
        x = self.fc(x)
        return x

# ============================================================================
# 4. DATA LOADING
# ============================================================================

def get_data_loaders(data_dir='./data', batch_size=128):
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256)
    return train_loader, test_loader, train_ds

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += x.size(0)
    return correct / total

# ============================================================================
# 5. STAGE 1: FP32 BASELINE TRAINING
# ============================================================================

def train_fp32(device, data_dir='./data', epochs=20, lr=1e-3, checkpoint='./tinynpu_fp32.pt'):
    print("=" * 60)
    print("Stage 1: FP32 Baseline Training")
    print("=" * 60)
    model = TinyNetFP32().to(device)
    train_loader, test_loader, _ = get_data_loaders(data_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        acc = evaluate(model, test_loader, device)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), checkpoint)
        print(f"  Epoch {epoch+1:2d}/{epochs}  lr={scheduler.get_last_lr()[0]:.6f}  Test Acc: {acc*100:.2f}%  (best: {best_acc*100:.2f}%)")

    print(f"\nFP32 training done. Best accuracy: {best_acc*100:.2f}%")
    print(f"Saved to {checkpoint}")
    return model

# ============================================================================
# 6. STAGE 2: POST-TRAINING QUANTIZATION (Calibration)
# ============================================================================

def calibrate_and_quantize(device, layer_configs, data_dir='./data',
                           fp32_checkpoint='./tinynpu_fp32.pt',
                           qat_checkpoint='./tinynpu_qat.pt',
                           num_calibration_samples=1000):
    print("=" * 60)
    print("Stage 2: Post-Training Quantization (Calibration)")
    print("=" * 60)

    # Load FP32 model
    fp32_model = TinyNetFP32().to(device)
    fp32_model.load_state_dict(torch.load(fp32_checkpoint, map_location=device, weights_only=True))
    fp32_model.eval()

    train_loader, test_loader, train_ds = get_data_loaders(data_dir)

    fp32_acc = evaluate(fp32_model, test_loader, device)
    print(f"  FP32 baseline accuracy: {fp32_acc*100:.2f}%")

    # Calibration: collect activation ranges per layer
    print(f"  Calibrating with {num_calibration_samples} samples...")
    calib_subset = Subset(train_ds, range(num_calibration_samples))
    calib_loader = DataLoader(calib_subset, batch_size=256)

    act_maxes = {'conv1': 0, 'conv2': 0, 'conv3': 0, 'fc': 0}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            x = input[0]
            act_maxes[name] = max(act_maxes[name], x.abs().max().item())
        return hook_fn

    hooks.append(fp32_model.conv1.register_forward_hook(make_hook('conv1')))
    hooks.append(fp32_model.conv2.register_forward_hook(make_hook('conv2')))
    hooks.append(fp32_model.conv3.register_forward_hook(make_hook('conv3')))
    hooks.append(fp32_model.fc.register_forward_hook(make_hook('fc')))

    with torch.no_grad():
        for x, _ in calib_loader:
            fp32_model(x.to(device))

    for h in hooks:
        h.remove()

    # Build QAT model with calibrated scales
    qat_model = TinyNetQAT(layer_configs).to(device)

    # Copy FP32 weights into QAT model
    fp32_sd = fp32_model.state_dict()
    qat_sd = qat_model.state_dict()
    weight_map = {
        'conv1.conv.weight': 'conv1.weight', 'conv1.conv.bias': 'conv1.bias',
        'conv2.conv.weight': 'conv2.weight', 'conv2.conv.bias': 'conv2.bias',
        'conv3.conv.weight': 'conv3.weight', 'conv3.conv.bias': 'conv3.bias',
        'fc.linear.weight': 'fc.weight', 'fc.linear.bias': 'fc.bias',
    }
    for qat_key, fp32_key in weight_map.items():
        qat_sd[qat_key] = fp32_sd[fp32_key]

    # Set calibrated activation scales
    layer_names = ['conv1', 'conv2', 'conv3', 'fc']
    for name in layer_names:
        a_bits = layer_configs[name]['a_bits']
        a_qmax = 2 ** a_bits - 1
        a_scale = act_maxes[name] / a_qmax if a_qmax > 0 else 1e-8
        a_scale = max(a_scale, 1e-8)
        qat_sd[f'{name}.a_scale'] = torch.tensor(a_scale)

        # Set weight scales
        w_bits = layer_configs[name]['w_bits']
        w_qmax = 2 ** (w_bits - 1) - 1
        if name == 'fc':
            w = fp32_sd['fc.weight']
        else:
            w = fp32_sd[f'{name}.weight']
        w_scale = (w.abs().max() / w_qmax).clamp(min=1e-8).item()
        qat_sd[f'{name}.w_scale'] = torch.tensor(w_scale)

    qat_model.load_state_dict(qat_sd)

    print("  Calibrated scales (intermediate FP, used to compute integer params):")
    for name in layer_names:
        layer = getattr(qat_model, name)
        w_scale = layer.w_scale.item()
        a_scale = layer.a_scale.item()
        # Show what the integer export values will be
        w_qmax = 2 ** (layer.w_bits - 1) - 1
        mod = layer.conv if isinstance(layer, QConv2d) else layer.linear
        w_int_max = int(np.clip(np.round(mod.weight.detach().cpu().numpy() / w_scale),
                                -(2**(layer.w_bits-1))+1, w_qmax).max())
        w_int_min = int(np.clip(np.round(mod.weight.detach().cpu().numpy() / w_scale),
                                -(2**(layer.w_bits-1))+1, w_qmax).min())
        b_int = np.round(mod.bias.detach().cpu().numpy() / (w_scale * a_scale)).astype(np.int32)
        print(f"    {name} (w{layer.w_bits}a{layer.a_bits}): "
              f"weight_int range=[{w_int_min}, {w_int_max}]  "
              f"bias_int32 range=[{b_int.min()}, {b_int.max()}]")

    # Evaluate PTQ accuracy
    ptq_acc = evaluate(qat_model, test_loader, device)
    print(f"\n  PTQ accuracy: {ptq_acc*100:.2f}%  (FP32: {fp32_acc*100:.2f}%, drop: {(fp32_acc-ptq_acc)*100:.2f}%)")

    torch.save(qat_model.state_dict(), qat_checkpoint)
    print(f"  Saved PTQ model to {qat_checkpoint}")
    return qat_model

# ============================================================================
# 7. STAGE 3: QAT FINE-TUNING
# ============================================================================

def finetune_qat(device, layer_configs, data_dir='./data',
                 qat_checkpoint='./tinynpu_qat.pt',
                 epochs=10, lr=1e-4):
    print("=" * 60)
    print("Stage 3: QAT Fine-Tuning")
    print("=" * 60)

    qat_model = TinyNetQAT(layer_configs).to(device)
    qat_model.load_state_dict(torch.load(qat_checkpoint, map_location=device, weights_only=True))

    train_loader, test_loader, _ = get_data_loaders(data_dir)

    pre_acc = evaluate(qat_model, test_loader, device)
    print(f"  Pre-finetune accuracy: {pre_acc*100:.2f}%")

    # Only optimize weights and biases, not scales
    params_to_optimize = []
    for name, param in qat_model.named_parameters():
        if 'scale' not in name:
            params_to_optimize.append(param)

    optimizer = torch.optim.Adam(params_to_optimize, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = pre_acc
    for epoch in range(epochs):
        qat_model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(qat_model(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        acc = evaluate(qat_model, test_loader, device)
        if acc > best_acc:
            best_acc = acc
            torch.save(qat_model.state_dict(), qat_checkpoint)
        print(f"  Epoch {epoch+1:2d}/{epochs}  lr={scheduler.get_last_lr()[0]:.6f}  Test Acc: {acc*100:.2f}%  (best: {best_acc*100:.2f}%)")

    print(f"\nQAT fine-tuning done. Best accuracy: {best_acc*100:.2f}%")
    print(f"Saved to {qat_checkpoint}")
    return qat_model

# ============================================================================
# 8. SENSITIVITY ANALYSIS
# ============================================================================

def run_sensitivity_analysis(device, data_dir='./data',
                             fp32_checkpoint='./tinynpu_fp32.pt',
                             qat_checkpoint='./tinynpu_qat.pt'):
    print("=" * 60)
    print("Sensitivity Analysis")
    print("=" * 60)

    _, test_loader, train_ds = get_data_loaders(data_dir)

    # Load FP32 model for calibration
    fp32_model = TinyNetFP32().to(device)
    fp32_model.load_state_dict(torch.load(fp32_checkpoint, map_location=device, weights_only=True))
    fp32_model.eval()

    fp32_acc = evaluate(fp32_model, test_loader, device)
    print(f"  FP32 baseline: {fp32_acc*100:.2f}%")

    # Evaluate INT8 baseline (all layers at W8A8)
    all8_configs = {k: {'w_bits': 8, 'a_bits': 8} for k in ['conv1', 'conv2', 'conv3', 'fc']}
    # Quick calibration helper
    def calibrate_quick(configs):
        """Calibrate and return PTQ model."""
        calib_loader = DataLoader(Subset(train_ds, range(1000)), batch_size=256)
        act_maxes = {'conv1': 0, 'conv2': 0, 'conv3': 0, 'fc': 0}
        hooks = []
        def make_hook(name):
            def hook_fn(module, input, output):
                act_maxes[name] = max(act_maxes[name], input[0].abs().max().item())
            return hook_fn
        hooks.append(fp32_model.conv1.register_forward_hook(make_hook('conv1')))
        hooks.append(fp32_model.conv2.register_forward_hook(make_hook('conv2')))
        hooks.append(fp32_model.conv3.register_forward_hook(make_hook('conv3')))
        hooks.append(fp32_model.fc.register_forward_hook(make_hook('fc')))
        with torch.no_grad():
            for x, _ in calib_loader:
                fp32_model(x.to(device))
        for h in hooks:
            h.remove()

        qat_model = TinyNetQAT(configs).to(device)
        fp32_sd = fp32_model.state_dict()
        qat_sd = qat_model.state_dict()
        weight_map = {
            'conv1.conv.weight': 'conv1.weight', 'conv1.conv.bias': 'conv1.bias',
            'conv2.conv.weight': 'conv2.weight', 'conv2.conv.bias': 'conv2.bias',
            'conv3.conv.weight': 'conv3.weight', 'conv3.conv.bias': 'conv3.bias',
            'fc.linear.weight': 'fc.weight', 'fc.linear.bias': 'fc.bias',
        }
        for qk, fk in weight_map.items():
            qat_sd[qk] = fp32_sd[fk]
        for name in ['conv1', 'conv2', 'conv3', 'fc']:
            a_bits = configs[name]['a_bits']
            a_qmax = 2 ** a_bits - 1
            qat_sd[f'{name}.a_scale'] = torch.tensor(max(act_maxes[name] / a_qmax, 1e-8))
            w_bits = configs[name]['w_bits']
            w_qmax = 2 ** (w_bits - 1) - 1
            fk = 'fc.weight' if name == 'fc' else f'{name}.weight'
            qat_sd[f'{name}.w_scale'] = torch.tensor(max((fp32_sd[fk].abs().max() / w_qmax).item(), 1e-8))
        qat_model.load_state_dict(qat_sd)
        return qat_model

    int8_model = calibrate_quick(all8_configs)
    int8_acc = evaluate(int8_model, test_loader, device)
    print(f"  INT8 baseline (all W8A8): {int8_acc*100:.2f}%")

    layers = ['conv1', 'conv2', 'conv3', 'fc']
    print(f"\n  Dropping each layer to W4A4 (others at W8A8):")
    importances = {}
    for layer_to_drop in layers:
        configs = {l: {'w_bits': 4, 'a_bits': 4} if l == layer_to_drop
                   else {'w_bits': 8, 'a_bits': 8} for l in layers}
        test_model = calibrate_quick(configs)
        acc = evaluate(test_model, test_loader, device)
        loss = int8_acc - acc
        importances[layer_to_drop] = loss
        print(f"    {layer_to_drop:6s} -> W4A4: {acc*100:5.2f}%  (drop: {loss*100:+5.2f}%)")

    sorted_layers = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Sensitivity ranking (most sensitive first):")
    for i, (name, loss) in enumerate(sorted_layers):
        recommendation = "keep INT8" if loss > 0.01 else "safe for INT4"
        print(f"    {i+1}. {name:6s}: drop={loss*100:+.2f}%  -> {recommendation}")

# ============================================================================
# 9. EXPORT FOR NPU
# ============================================================================

def export_for_npu(device, layer_configs, qat_checkpoint='./tinynpu_qat.pt',
                   export_dir='./mnist_mixed_export'):
    print("=" * 60)
    print("Export for NPU")
    print("=" * 60)

    os.makedirs(export_dir, exist_ok=True)

    model = TinyNetQAT(layer_configs).to(device)
    model.load_state_dict(torch.load(qat_checkpoint, map_location=device, weights_only=True))
    model.eval()

    layers_info = []
    named_layers = [
        ('conv1', model.conv1),
        ('conv2', model.conv2),
        ('conv3', model.conv3),
        ('fc', model.fc),
    ]

    for i, (name, layer) in enumerate(named_layers):
        is_conv = isinstance(layer, QConv2d)
        mod = layer.conv if is_conv else layer.linear
        w_scale = layer.w_scale.item()
        a_scale = layer.a_scale.item()

        # Output scale = next layer's activation scale, or self for last layer
        if i < len(named_layers) - 1:
            out_scale = named_layers[i + 1][1].a_scale.item()
        else:
            out_scale = w_scale * a_scale

        # Quantize weights
        w_qmax = 2 ** (layer.w_bits - 1) - 1
        w_qmin = -(2 ** (layer.w_bits - 1)) + 1
        w_int = np.clip(
            np.round(mod.weight.detach().cpu().numpy() / w_scale),
            w_qmin, w_qmax
        ).astype(np.int32)

        # Quantize biases (32-bit, scaled by w_scale * a_scale)
        b_int = np.round(mod.bias.detach().cpu().numpy() / (w_scale * a_scale)).astype(np.int32)

        # Fused requantization params
        M0, shift = compute_fused_params(w_scale, a_scale, out_scale)

        layer_info = {
            'name': name,
            'type': 'conv2d' if is_conv else 'linear',
            'w_bits': layer.w_bits,
            'a_bits': layer.a_bits,
            'M0': int(M0),
            'shift': int(shift),
        }

        if is_conv:
            layer_info.update({
                'kernel_size': mod.kernel_size[0],
                'padding': mod.padding[0],
                'stride': mod.stride[0],
                'in_channels': mod.in_channels,
                'out_channels': mod.out_channels,
            })
            np.save(os.path.join(export_dir, f'{name}_weights_gemm.npy'),
                    w_int.reshape(mod.out_channels, -1))
        else:
            layer_info.update({
                'in_features': mod.in_features,
                'out_features': mod.out_features,
            })
            np.save(os.path.join(export_dir, f'{name}_weights.npy'), w_int)

        np.save(os.path.join(export_dir, f'{name}_bias.npy'), b_int)
        layers_info.append(layer_info)
        print(f"  {name}: W{layer.w_bits}A{layer.a_bits}  M0={M0}  shift={shift}  "
              f"w_scale={w_scale:.6f}  a_scale={a_scale:.6f}")

    manifest = {
        'input_scale': float(named_layers[0][1].a_scale.item()),
        'layers': layers_info,
    }
    with open(os.path.join(export_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Exported to {export_dir}/")
    print(f"  manifest.json + {len(layers_info)} layers")

# ============================================================================
# 10. MAIN
# ============================================================================

def parse_layer_configs(args):
    return {
        'conv1': {'w_bits': args.conv1_bits[0], 'a_bits': args.conv1_bits[1]},
        'conv2': {'w_bits': args.conv2_bits[0], 'a_bits': args.conv2_bits[1]},
        'conv3': {'w_bits': args.conv3_bits[0], 'a_bits': args.conv3_bits[1]},
        'fc':    {'w_bits': args.fc_bits[0],    'a_bits': args.fc_bits[1]},
    }

def main():
    parser = argparse.ArgumentParser(description='TinyNPU: Train FP32 -> Quantize -> Fine-Tune -> Export')

    # Stage selection
    parser.add_argument('--train', action='store_true', help='Stage 1: FP32 baseline training')
    parser.add_argument('--quantize', action='store_true', help='Stage 2: PTQ calibration + eval')
    parser.add_argument('--finetune', action='store_true', help='Stage 3: QAT fine-tuning')
    parser.add_argument('--sensitivity', action='store_true', help='Sensitivity analysis')
    parser.add_argument('--export', action='store_true', help='Export quantized model for NPU')

    # Paths
    parser.add_argument('--fp32-checkpoint', type=str, default='./tinynpu_fp32.pt')
    parser.add_argument('--qat-checkpoint', type=str, default='./tinynpu_qat.pt')
    parser.add_argument('--export-dir', type=str, default='./mnist_mixed_export')
    parser.add_argument('--data-dir', type=str, default='./data')

    # Training params
    parser.add_argument('--fp32-epochs', type=int, default=50)
    parser.add_argument('--fp32-lr', type=float, default=1e-3)
    parser.add_argument('--qat-epochs', type=int, default=10)
    parser.add_argument('--qat-lr', type=float, default=1e-4)

    # Per-layer bit-width: --conv1-bits W A
    parser.add_argument('--conv1-bits', nargs=2, type=int, default=[8, 8], metavar=('W', 'A'))
    parser.add_argument('--conv2-bits', nargs=2, type=int, default=[8, 8], metavar=('W', 'A'))
    parser.add_argument('--conv3-bits', nargs=2, type=int, default=[8, 8], metavar=('W', 'A'))
    parser.add_argument('--fc-bits', nargs=2, type=int, default=[8, 8], metavar=('W', 'A'))

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    layer_configs = parse_layer_configs(args)

    ran_something = False

    if args.train:
        ran_something = True
        train_fp32(device, data_dir=args.data_dir, epochs=args.fp32_epochs,
                   lr=args.fp32_lr, checkpoint=args.fp32_checkpoint)

    if args.quantize:
        ran_something = True
        calibrate_and_quantize(device, layer_configs, data_dir=args.data_dir,
                               fp32_checkpoint=args.fp32_checkpoint,
                               qat_checkpoint=args.qat_checkpoint)

    if args.finetune:
        ran_something = True
        finetune_qat(device, layer_configs, data_dir=args.data_dir,
                     qat_checkpoint=args.qat_checkpoint,
                     epochs=args.qat_epochs, lr=args.qat_lr)

    if args.sensitivity:
        ran_something = True
        run_sensitivity_analysis(device, data_dir=args.data_dir,
                                 fp32_checkpoint=args.fp32_checkpoint,
                                 qat_checkpoint=args.qat_checkpoint)

    if args.export:
        ran_something = True
        export_for_npu(device, layer_configs, qat_checkpoint=args.qat_checkpoint,
                       export_dir=args.export_dir)

    if not ran_something:
        parser.print_help()
        print("\nExample workflow:")
        print("  python quant-by-claude.py --train")
        print("  python quant-by-claude.py --quantize")
        print("  python quant-by-claude.py --sensitivity")
        print("  python quant-by-claude.py --finetune --conv2-bits 4 4")
        print("  python quant-by-claude.py --export --export-dir ./mnist_mixed_export")

if __name__ == '__main__':
    main()
