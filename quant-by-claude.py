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
import numpy as np
import json
import os
import argparse

from software.compiler.tinynpu_quant import compute_fused_params
from software.compiler.tinynpu_quant import (
    QConv2d,
    QLinear,
    build_layer_config_map,
    collect_input_activation_maxes,
    copy_state_with_mapping,
    initialize_scale_tensors,
    rank_sensitivity,
    single_layer_bit_drop_sensitivity,
)

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

class TinyNetQAT(nn.Module):
    """Quantization-aware model, initialized from FP32 weights + calibrated scales."""
    def __init__(self, layer_configs=None):
        super().__init__()
        c = build_layer_config_map(['conv1', 'conv2', 'conv3', 'fc'], overrides=layer_configs)
        self.conv1 = QConv2d(1, 16, 3, padding=1, config=c['conv1'])
        self.conv2 = QConv2d(16, 16, 3, padding=1, config=c['conv2'])
        self.conv3 = QConv2d(16, 16, 3, padding=1, config=c['conv3'])
        self.fc = QLinear(16, 10, config=c['fc'])

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
    from torchvision import datasets, transforms

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

    layer_names = ['conv1', 'conv2', 'conv3', 'fc']
    layer_configs = build_layer_config_map(layer_names, overrides=layer_configs)

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

    act_maxes = collect_input_activation_maxes(fp32_model, calib_loader, layer_names, device=device)

    # Build QAT model with calibrated scales
    qat_model = TinyNetQAT(layer_configs).to(device)

    fp32_sd = fp32_model.state_dict()
    qat_sd = qat_model.state_dict()
    weight_map = {
        'conv1.conv.weight': 'conv1.weight', 'conv1.conv.bias': 'conv1.bias',
        'conv2.conv.weight': 'conv2.weight', 'conv2.conv.bias': 'conv2.bias',
        'conv3.conv.weight': 'conv3.weight', 'conv3.conv.bias': 'conv3.bias',
        'fc.linear.weight': 'fc.weight', 'fc.linear.bias': 'fc.bias',
    }
    copy_state_with_mapping(dst_state_dict=qat_sd, src_state_dict=fp32_sd, key_mapping=weight_map)
    initialize_scale_tensors(
        qat_state_dict=qat_sd,
        fp32_state_dict=fp32_sd,
        layer_configs=layer_configs,
        activation_maxes=act_maxes,
        fp32_weight_keys={
            'conv1': 'conv1.weight',
            'conv2': 'conv2.weight',
            'conv3': 'conv3.weight',
            'fc': 'fc.weight',
        },
    )

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
    layers = ['conv1', 'conv2', 'conv3', 'fc']
    all8_configs = build_layer_config_map(layers)
    # Quick calibration helper
    def calibrate_quick(configs):
        """Calibrate and return PTQ model."""
        calib_loader = DataLoader(Subset(train_ds, range(1000)), batch_size=256)
        act_maxes = collect_input_activation_maxes(fp32_model, calib_loader, layers, device=device)

        qat_model = TinyNetQAT(configs).to(device)
        fp32_sd = fp32_model.state_dict()
        qat_sd = qat_model.state_dict()
        weight_map = {
            'conv1.conv.weight': 'conv1.weight', 'conv1.conv.bias': 'conv1.bias',
            'conv2.conv.weight': 'conv2.weight', 'conv2.conv.bias': 'conv2.bias',
            'conv3.conv.weight': 'conv3.weight', 'conv3.conv.bias': 'conv3.bias',
            'fc.linear.weight': 'fc.weight', 'fc.linear.bias': 'fc.bias',
        }
        copy_state_with_mapping(dst_state_dict=qat_sd, src_state_dict=fp32_sd, key_mapping=weight_map)
        initialize_scale_tensors(
            qat_state_dict=qat_sd,
            fp32_state_dict=fp32_sd,
            layer_configs=configs,
            activation_maxes=act_maxes,
            fp32_weight_keys={
                'conv1': 'conv1.weight',
                'conv2': 'conv2.weight',
                'conv3': 'conv3.weight',
                'fc': 'fc.weight',
            },
        )
        qat_model.load_state_dict(qat_sd)
        return qat_model

    int8_model = calibrate_quick(all8_configs)
    int8_acc = evaluate(int8_model, test_loader, device)
    print(f"  INT8 baseline (all W8A8): {int8_acc*100:.2f}%")

    print(f"\n  Dropping each layer to W4A4 (others at W8A8):")
    def evaluate_configs(configs):
        return evaluate(calibrate_quick(configs), test_loader, device)

    _, importances = single_layer_bit_drop_sensitivity(
        layers,
        evaluate_configs=evaluate_configs,
        baseline_bits=(8, 8),
        trial_bits=(4, 4),
    )
    for layer_to_drop, loss in importances.items():
        print(f"    {layer_to_drop:6s} -> W4A4: {(int8_acc - loss)*100:5.2f}%  (drop: {loss*100:+5.2f}%)")

    sorted_layers = rank_sensitivity(importances)
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
    return build_layer_config_map(
        ['conv1', 'conv2', 'conv3', 'fc'],
        overrides={
            'conv1': {'w_bits': args.conv1_bits[0], 'a_bits': args.conv1_bits[1]},
            'conv2': {'w_bits': args.conv2_bits[0], 'a_bits': args.conv2_bits[1]},
            'conv3': {'w_bits': args.conv3_bits[0], 'a_bits': args.conv3_bits[1]},
            'fc': {'w_bits': args.fc_bits[0], 'a_bits': args.fc_bits[1]},
        },
    )

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
