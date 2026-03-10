from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import Any

import numpy as np

from .golden import GoldenModel
from .ir import DType, ExecutionPlan, HostOp, MatMulOp, NpuSegment, TensorKind, TensorSpec, VerifyTensor, normalize_shape


@dataclass
class PartitionState:
    tensors: dict[str, TensorSpec]
    steps: list[Any]
    inputs: list[str]
    outputs: list[str]
    expected_tensors: dict[str, np.ndarray]


def partition_fx_graph(graph_module: Any, example_inputs: tuple[Any, ...], verify_policy: str | None = None, **_: Any):
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        raise ImportError("torch is required for FX partitioning.") from exc

    tensors: dict[str, TensorSpec] = {}
    steps: list[Any] = []
    expected_tensors: dict[str, np.ndarray] = {}
    inputs: list[str] = []
    outputs: list[str] = []
    current_ops: list[MatMulOp] = []
    current_inputs: set[str] = set()

    env: dict[str, np.ndarray] = {}
    example_iter = iter(example_inputs)
    golden = GoldenModel()

    def to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.array(value)

    def parse_dtype(name: str) -> DType:
        normalized = str(name).lower()
        mapping = {
            "int4": DType.INT4,
            "int8": DType.INT8,
            "int16": DType.INT16,
            "int32": DType.INT32,
            "float32": DType.FLOAT32,
        }
        if normalized not in mapping:
            raise ValueError(f"Unsupported dtype annotation {name!r}.")
        return mapping[normalized]

    def coerce_npu_tensor(name: str) -> np.ndarray:
        value = np.array(env[name], copy=False)
        if np.issubdtype(value.dtype, np.integer):
            return value.astype(np.int16, copy=False)
        rounded = np.rint(value)
        if np.allclose(value, rounded, rtol=0.0, atol=1e-6):
            coerced = rounded.astype(np.int16)
            env[name] = coerced
            if name in tensors and tensors[name].dtype == DType.FLOAT32:
                tensors[name].dtype = DType.INT16
                if tensors[name].data is not None:
                    tensors[name].data = np.array(coerced, copy=True)
            return coerced
        raise NotImplementedError(
            f"Tensor '{name}' is floating-point at an NPU boundary. "
            "Insert quantize_for_npu(...) before feeding it into a TinyNPU segment."
        )

    modules = dict(graph_module.named_modules())

    def flush_segment() -> None:
        nonlocal current_ops, current_inputs
        if not current_ops:
            return
        outputs_local = [current_ops[-1].out]
        steps.append(
            NpuSegment(
                name=f"segment_{len([s for s in steps if isinstance(s, NpuSegment)]):03d}",
                ops=list(current_ops),
                inputs=sorted(current_inputs),
                outputs=outputs_local,
            )
        )
        current_ops = []
        current_inputs = set()
    for node in graph_module.graph.nodes:
        if node.op == "placeholder":
            value = to_numpy(next(example_iter))
            env[node.name] = value
            tensors[node.name] = TensorSpec(
                name=node.name,
                shape=normalize_shape(value.shape),
                dtype=DType.INT16,
                kind=TensorKind.INPUT,
            )
            inputs.append(node.name)
            continue

        if node.op == "get_attr":
            target = graph_module
            for part in str(node.target).split("."):
                target = getattr(target, part)
            value = to_numpy(target)
            env[node.name] = value
            tensors[node.name] = TensorSpec(
                name=node.name,
                shape=normalize_shape(value.shape),
                dtype=DType.INT16 if np.issubdtype(value.dtype, np.integer) else DType.FLOAT32,
                kind=TensorKind.CONSTANT,
                data=value.astype(np.int16) if np.issubdtype(value.dtype, np.integer) else value,
            )
            continue

        if node.op == "call_module" and isinstance(modules[node.target], nn.Linear):
            module = modules[node.target]
            inp_node = node.args[0]
            lhs_name = f"{node.name}_weight"
            rhs_name = inp_node.name
            out_name = node.name
            bias_name = None
            weight = module.weight.detach().cpu().numpy().astype(np.int16)
            tensors[lhs_name] = TensorSpec(lhs_name, normalize_shape(weight.shape), DType.INT16, TensorKind.CONSTANT, data=weight)
            current_inputs.update({lhs_name, rhs_name})
            if module.bias is not None:
                bias = module.bias.detach().cpu().numpy().reshape(1, -1).astype(np.int32)
                bias_name = f"{node.name}_bias"
                tensors[bias_name] = TensorSpec(bias_name, normalize_shape(bias.shape), DType.INT32, TensorKind.CONSTANT, data=bias)
                current_inputs.add(bias_name)
            rhs = coerce_npu_tensor(rhs_name)
            expected = golden.matmul(weight, rhs, bias=tensors[bias_name].data if bias_name else None)
            env[out_name] = expected.astype(np.int32)
            tensors[out_name] = TensorSpec(out_name, normalize_shape(expected.shape), DType.INT16, TensorKind.INTERMEDIATE)
            current_ops.append(MatMulOp(name=node.name, lhs=lhs_name, rhs=rhs_name, out=out_name, bias=bias_name))
            continue

        if (
            node.op == "call_function"
            and node.target in {operator.matmul, getattr(torch, "matmul", None)}
        ) or (node.op == "call_method" and node.target == "matmul"):
            lhs_node = node.args[0]
            rhs_node = node.args[1]
            lhs_name = lhs_node.name
            rhs_name = rhs_node.name
            out_name = node.name
            current_inputs.update({lhs_name, rhs_name})
            lhs_value = coerce_npu_tensor(lhs_name)
            rhs_value = coerce_npu_tensor(rhs_name)
            expected = golden.matmul(lhs_value, rhs_value)
            env[out_name] = expected.astype(np.int32)
            tensors[out_name] = TensorSpec(out_name, normalize_shape(expected.shape), DType.INT16, TensorKind.INTERMEDIATE)
            current_ops.append(MatMulOp(name=node.name, lhs=lhs_name, rhs=rhs_name, out=out_name))
            continue

        if node.op == "call_method" and node.target == "reshape":
            source = node.args[0].name
            shape = tuple(int(dim) for dim in node.args[1:])
            env[node.name] = np.reshape(env[source], shape)
            tensors[node.name] = TensorSpec(
                name=node.name,
                shape=normalize_shape(env[node.name].shape),
                dtype=tensors[source].dtype,
                kind=tensors[source].kind,
                data=np.array(env[node.name], copy=True) if tensors[source].kind == TensorKind.CONSTANT else None,
            )
            continue

        if node.op == "call_function" and node.target in {operator.add, getattr(torch, "add", None)}:
            lhs_node = node.args[0]
            rhs_node = node.args[1]
            lhs_name = lhs_node.name
            rhs_name = rhs_node.name
            if current_ops and current_ops[-1].out == lhs_name and tensors[rhs_name].kind == TensorKind.CONSTANT:
                bias_value = np.array(env[rhs_name], copy=True).reshape(1, -1).astype(np.int32)
                bias_name = f"{node.name}_bias"
                tensors[bias_name] = TensorSpec(
                    name=bias_name,
                    shape=normalize_shape(bias_value.shape),
                    dtype=DType.INT32,
                    kind=TensorKind.CONSTANT,
                    data=bias_value,
                )
                current_inputs.add(bias_name)
                matmul_op = current_ops[-1]
                matmul_op.bias = bias_name
                matmul_op.out = node.name
                expected = golden.matmul(env[matmul_op.lhs], env[matmul_op.rhs], bias=bias_value)
                env[node.name] = expected.astype(np.int32)
                tensors[node.name] = TensorSpec(node.name, normalize_shape(expected.shape), DType.INT16, TensorKind.INTERMEDIATE)
                continue
            raise NotImplementedError(
                "Only constant bias-add immediately following a matmul is currently supported."
            )

        if node.op == "call_function" and getattr(node.target, "__name__", None) == "mark_for_verify":
            flush_segment()
            source = node.args[0].name
            label = node.args[1] if len(node.args) > 1 else None
            env[node.name] = env[source]
            tensors[node.name] = tensors[source].clone_without_data(name=node.name)
            tensors[node.name].verify_label = label or node.name
            steps.append(HostOp(name=f"{node.name}_alias", kind="alias", inputs=[source], outputs=[node.name]))
            steps.append(
                VerifyTensor(
                    tensor_name=node.name,
                    label=tensors[node.name].verify_label,
                    is_final_output=tensors[node.name].is_final_output,
                )
            )
            expected_tensors[node.name] = np.array(env[node.name], copy=True)
            continue

        if node.op == "call_function" and getattr(node.target, "__name__", None) == "softmax":
            flush_segment()
            source = node.args[0].name
            axis = int(node.kwargs.get("dim", -1))
            out_name = node.name
            steps.append(HostOp(name=node.name, kind="softmax", inputs=[source], outputs=[out_name], attrs={"axis": axis}))
            source_arr = env[source].astype(np.float32)
            shifted = source_arr - np.max(source_arr, axis=axis, keepdims=True)
            exp = np.exp(shifted)
            env[out_name] = exp / np.sum(exp, axis=axis, keepdims=True)
            tensors[out_name] = TensorSpec(out_name, normalize_shape(env[out_name].shape), DType.FLOAT32, TensorKind.INTERMEDIATE)
            expected_tensors[out_name] = np.array(env[out_name], copy=True)
            continue

        if node.op == "call_function" and getattr(node.target, "__name__", None) == "quantize_for_npu":
            flush_segment()
            source = node.args[0].name
            scale = float(node.args[1])
            zero_point = int(node.args[2]) if len(node.args) > 2 else 0
            dtype = parse_dtype(node.args[3] if len(node.args) > 3 else "int16")
            out_name = node.name
            steps.append(
                HostOp(
                    name=node.name,
                    kind="requantize",
                    inputs=[source],
                    outputs=[out_name],
                    attrs={"scale": scale, "zero_point": zero_point, "dtype": dtype},
                )
            )
            env[out_name] = golden.requantize(env[source], scale=scale, zero_point=zero_point, out_dtype=dtype)
            tensors[out_name] = TensorSpec(
                out_name,
                normalize_shape(env[out_name].shape),
                dtype,
                TensorKind.INTERMEDIATE,
                metadata={"quantization": {"scale": scale, "zero_point": zero_point, "dtype": dtype.value}},
            )
            expected_tensors[out_name] = np.array(env[out_name], copy=True)
            continue

        if node.op == "output":
            flush_segment()
            raw_outputs = node.args[0]
            output_nodes = raw_outputs if isinstance(raw_outputs, (tuple, list)) else (raw_outputs,)
            for out in output_nodes:
                tensors[out.name].is_final_output = True
                steps.append(VerifyTensor(tensor_name=out.name, label=tensors[out.name].verify_label or out.name, is_final_output=True))
                outputs.append(out.name)
                expected_tensors[out.name] = np.array(env[out.name], copy=True)
            continue

        raise NotImplementedError(
            f"Unsupported FX node for initial frontend: op={node.op}, target={node.target!r}."
        )

    for step in steps:
        if isinstance(step, NpuSegment):
            for out_name in step.outputs:
                expected_tensors[out_name] = np.array(env[out_name], copy=True)

    plan = ExecutionPlan(tensors=tensors, steps=steps, inputs=inputs, outputs=outputs, metadata={"frontend": "torch.fx"})
    return plan, expected_tensors
