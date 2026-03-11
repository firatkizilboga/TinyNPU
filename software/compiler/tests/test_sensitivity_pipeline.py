import os
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

TESTS_DIR = Path(__file__).resolve().parent
COMPILER_DIR = TESTS_DIR.parent
REPO_ROOT = COMPILER_DIR.parent.parent
sys.path.insert(0, str(COMPILER_DIR))
sys.path.insert(0, str(REPO_ROOT))

from tinynpu_jit import VerificationMode, compile_module, run_host_emulation
from tinynpu_quant import (
    LayerQuantConfig,
    QConv2d,
    QLinear,
    apply_layer_quant_configs,
    build_mixed_precision_sensitivity_report,
    convert_mixed_precision_qat_model_for_compiler,
    recalibrate_qat_scales,
)


class TinyMixedPrecisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = QConv2d(1, 2, 3, padding=1, config=LayerQuantConfig(w_bits=8, a_bits=8, signed_activations=True))
        self.fc = QLinear(2, 3, config=LayerQuantConfig(w_bits=8, a_bits=8, signed_activations=True))
        with torch.no_grad():
            self.conv.conv.weight.copy_(
                torch.tensor(
                    [
                        [[[0.50, -0.25, 0.125], [0.25, 0.75, -0.5], [0.0, 0.125, -0.25]]],
                        [[[-0.125, 0.375, 0.5], [0.25, -0.625, 0.25], [0.125, 0.0, -0.125]]],
                    ],
                    dtype=torch.float32,
                )
            )
            self.conv.conv.bias.copy_(torch.tensor([0.125, -0.25], dtype=torch.float32))
            self.fc.linear.weight.copy_(
                torch.tensor(
                    [
                        [0.75, -0.50],
                        [-0.25, 0.50],
                        [0.125, -0.375],
                    ],
                    dtype=torch.float32,
                )
            )
            self.fc.linear.bias.copy_(torch.tensor([0.125, -0.25, 0.375], dtype=torch.float32))
            self.conv.w_scale.copy_(torch.tensor(0.125))
            self.conv.a_scale.copy_(torch.tensor(0.125))
            self.fc.w_scale.copy_(torch.tensor(0.125))
            self.fc.a_scale.copy_(torch.tensor(0.125))

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = x.mean(dim=[2, 3])
        x = self.fc(x)
        return x


class SensitivityPipelineTests(unittest.TestCase):
    def test_report_ranks_layers_and_selects_lowest_acceptable_precision(self):
        layer_names = ["conv1", "conv2", "fc"]
        penalties = {
            ("conv1", 4, 4): 0.030,
            ("conv2", 4, 4): 0.004,
            ("fc", 4, 4): 0.009,
            ("conv1", 16, 16): -0.001,
            ("conv2", 16, 16): -0.001,
            ("fc", 16, 16): -0.001,
        }

        def evaluate_configs(configs):
            score = 0.970
            for layer_name, config in configs.items():
                if (config.w_bits, config.a_bits) != (8, 8):
                    score -= penalties.get((layer_name, config.w_bits, config.a_bits), 0.0)
            return score

        report = build_mixed_precision_sensitivity_report(
            layer_names,
            evaluate_configs=evaluate_configs,
            candidate_bits=((16, 16), (8, 8), (4, 4)),
            max_acceptable_drop=0.010,
            parameter_counts={
                "conv1": {"weight_elements": 144, "bias_elements": 16},
                "conv2": {"weight_elements": 288, "bias_elements": 16},
                "fc": {"weight_elements": 160, "bias_elements": 10},
            },
        )

        self.assertEqual(report["selected_layer_configs"]["conv1"]["w_bits"], 8)
        self.assertEqual(report["selected_layer_configs"]["conv2"]["w_bits"], 4)
        self.assertEqual(report["selected_layer_configs"]["fc"]["w_bits"], 4)
        self.assertEqual(
            [entry["layer_name"] for entry in report["sensitivity_ranking"]],
            ["conv1", "fc", "conv2"],
        )
        self.assertLess(report["estimated_weight_bytes"]["selected_total"], report["estimated_weight_bytes"]["baseline_total"])
        conv1_entry = next(entry for entry in report["layers"] if entry["layer_name"] == "conv1")
        self.assertEqual(conv1_entry["recommendation"], "keep_w8a8")
        self.assertEqual(conv1_entry["rank"], 1)
        conv2_entry = next(entry for entry in report["layers"] if entry["layer_name"] == "conv2")
        self.assertEqual(conv2_entry["recommendation"], "use_w4a4")
        self.assertTrue(report["fine_tuning_recommended"])

    def test_apply_layer_quant_configs_keeps_original_model_unchanged(self):
        model = TinyMixedPrecisionModel().eval()
        updated = apply_layer_quant_configs(
            model,
            {
                "conv": LayerQuantConfig(w_bits=4, a_bits=4, signed_activations=True),
                "fc": LayerQuantConfig(w_bits=16, a_bits=16, signed_activations=True),
            },
            inplace=False,
        )

        self.assertEqual((model.conv.w_bits, model.conv.a_bits), (8, 8))
        self.assertEqual((model.fc.w_bits, model.fc.a_bits), (8, 8))
        self.assertEqual((updated.conv.w_bits, updated.conv.a_bits), (4, 4))
        self.assertEqual((updated.fc.w_bits, updated.fc.a_bits), (16, 16))

    def test_selected_policy_converts_and_compiles(self):
        model = TinyMixedPrecisionModel().eval()
        compiler_ready = convert_mixed_precision_qat_model_for_compiler(
            model,
            {
                "conv": LayerQuantConfig(w_bits=4, a_bits=4, signed_activations=True),
                "fc": LayerQuantConfig(w_bits=16, a_bits=16, signed_activations=True),
            },
            layer_order=["conv", "fc"],
        )

        self.assertEqual(compiler_ready.inner.conv.in_dtype, "int4")
        self.assertEqual(compiler_ready.inner.conv.out_dtype, "int16")
        self.assertEqual(compiler_ready.inner.fc.in_dtype, "int16")
        self.assertEqual(compiler_ready.inner.fc.out_dtype, "int16")

        example = torch.tensor(
            [[[[0.25, -0.5, 0.75], [0.125, -0.25, 0.5], [0.0, 0.375, -0.125]]]],
            dtype=torch.float32,
        )
        artifact = compile_module(compiler_ready, (example,))
        result = run_host_emulation(
            artifact,
            {artifact.plan.inputs[0]: example.numpy()},
            VerificationMode.OFF,
            debug=False,
        )

        output = result.tensors[artifact.plan.outputs[0]]
        self.assertIn(artifact.plan.outputs[0], result.tensors)
        self.assertGreater(output.size, 0)
        self.assertTrue(torch.isfinite(torch.from_numpy(output)).all().item())

    def test_report_uses_prepare_hook_for_candidate_evaluation(self):
        layer_names = ["conv1", "conv2"]
        prepared_calls = []

        def prepare_configs(configs):
            prepared_calls.append({name: (cfg.w_bits, cfg.a_bits) for name, cfg in configs.items()})
            return configs

        def evaluate_configs(configs):
            return 1.0

        report = build_mixed_precision_sensitivity_report(
            layer_names,
            evaluate_configs=evaluate_configs,
            candidate_bits=((8, 8), (4, 4)),
            prepare_configs=prepare_configs,
        )

        self.assertEqual(report["baseline_score"], 1.0)
        self.assertGreaterEqual(len(prepared_calls), len(layer_names) * 2)

    def test_raw_layer_config_conversion_preserves_signed_activations_from_model(self):
        model = TinyMixedPrecisionModel().eval()
        self.assertTrue(model.conv.signed_activations)
        self.assertTrue(model.fc.signed_activations)

        compiler_ready = convert_mixed_precision_qat_model_for_compiler(
            model,
            {
                "conv": {"w_bits": 4, "a_bits": 4},
                "fc": {"w_bits": 16, "a_bits": 16},
            },
            layer_order=["conv", "fc"],
        )

        self.assertEqual(compiler_ready.inner.conv.in_dtype, "int4")
        self.assertEqual(compiler_ready.inner.fc.in_dtype, "int16")

    def test_recalibrate_qat_scales_updates_for_new_precision(self):
        model = TinyMixedPrecisionModel().eval()
        updated = apply_layer_quant_configs(
            model,
            {"conv": LayerQuantConfig(w_bits=4, a_bits=4, signed_activations=True)},
            inplace=False,
        )
        dataset = [
            (torch.tensor([[[[0.25, -0.5, 0.75], [0.125, -0.25, 0.5], [0.0, 0.375, -0.125]]]], dtype=torch.float32), 0),
            (torch.tensor([[[[0.5, -0.25, 0.125], [0.75, -0.5, 0.25], [0.125, 0.0, -0.125]]]], dtype=torch.float32), 1),
        ]

        before = float(updated.conv.a_scale.item())
        recalibrate_qat_scales(updated, layer_names=["conv", "fc"], calib_loader=dataset, device="cpu", inplace=True)
        after = float(updated.conv.a_scale.item())

        self.assertGreater(after, 0.0)
        self.assertNotEqual(before, after)


if __name__ == "__main__":
    unittest.main()
