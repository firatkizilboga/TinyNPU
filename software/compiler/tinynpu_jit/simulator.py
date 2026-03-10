from __future__ import annotations

from typing import Any

import numpy as np

from tinynpu import TinyNPUProgram
from tinynpu.isa import PrecisionMode

from .artifact import CompiledArtifact, ExecutionResult
from .executor import HostEmulationExecutor
from .ir import NpuSegment, TensorSpec, VerificationMode, VerifyTensor


class SimulatorExecutor:
    def __init__(self, defines_path: str | None = None):
        self.program = TinyNPUProgram(defines_path=defines_path)
        self.host_executor = HostEmulationExecutor()
        self.array_size = self.program.array_size
        self.buffer_width = self.program.buffer_width
        self.im_base_addr = self.program.im_base_addr
        self.packer = self.program.packer

    async def run(
        self,
        artifact: CompiledArtifact,
        inputs: dict[str, np.ndarray],
        *,
        dut: Any,
        verification: VerificationMode = VerificationMode.OFF,
        reset: bool = False,
        capture_vectors: bool = False,
    ) -> ExecutionResult:
        try:
            from verification.cocotb import npu_driver
            from cocotb.clock import Clock
            from cocotb.triggers import ClockCycles, RisingEdge
            import cocotb
        except Exception as exc:
            raise ImportError(
                "Simulator execution requires the cocotb driver modules from verification/cocotb."
            ) from exc

        if reset:
            clock = Clock(dut.clk, 10, units="ns")
            cocotb.start_soon(clock.start())
            dut.rst_n.value = 0
            await ClockCycles(dut.clk, 5)
            dut.rst_n.value = 1

        values: dict[str, np.ndarray] = {}
        for name, spec in artifact.plan.tensors.items():
            if spec.data is not None:
                values[name] = np.array(spec.data, copy=True)
        for name in artifact.plan.inputs:
            if name not in inputs:
                raise KeyError(f"Missing runtime input '{name}'.")
            values[name] = np.array(inputs[name], copy=True)

        verified: list[str] = []
        vector_captures: dict[str, dict[str, Any]] = {}
        for step in artifact.plan.steps:
            if isinstance(step, NpuSegment):
                segment = artifact.segment_artifacts[step.name]
                await self._load_ub_image(dut, npu_driver, segment.binary["ub"])
                await self._overlay_runtime_inputs(dut, npu_driver, artifact, segment.symbol_table, step.inputs, values)
                await self._load_im_image(dut, npu_driver, segment.binary["im"])
                await self._run_until_halt(dut, npu_driver, RisingEdge)
                for output_name in step.outputs:
                    symbol = segment.symbol_table[output_name]
                    if capture_vectors:
                        vector_captures[output_name] = await self._capture_tensor_vectors(
                            dut,
                            npu_driver,
                            segment_name=step.name,
                            symbol=symbol,
                        )
                    values[output_name] = await self._read_tensor(dut, npu_driver, artifact.plan.tensors[output_name], symbol)
            elif step.__class__.__name__ == "HostOp":
                self.host_executor._run_host_op(step, values)
            elif isinstance(step, VerifyTensor):
                if self.host_executor._should_verify(step, verification):
                    expected = artifact.expected_tensors[step.tensor_name]
                    actual = values[step.tensor_name]
                    if np.issubdtype(actual.dtype, np.floating) or np.issubdtype(expected.dtype, np.floating):
                        matches = np.allclose(actual, expected, rtol=1e-5, atol=1e-6)
                    else:
                        matches = np.array_equal(actual, expected)
                    if not matches:
                        raise AssertionError(f"Verification failed for '{step.label}' ({step.tensor_name}).")
                    verified.append(step.label)

        outputs = {name: np.array(values[name], copy=True) for name in artifact.plan.outputs}
        trace_tensors = {name: np.array(value, copy=True) for name, value in values.items()}
        return ExecutionResult(
            tensors=outputs,
            verified=verified,
            trace_tensors=trace_tensors,
            vector_captures=vector_captures,
        )

    async def _load_ub_image(self, dut: Any, driver: Any, ub_words: list[int]) -> None:
        for addr, word in enumerate(ub_words):
            await driver.write_reg(dut, driver.REG_ADDR, addr, 16)
            await driver.write_reg(dut, driver.REG_CMD, 0x01, 8)
            await driver.write_reg(dut, driver.REG_MMVR, word, self.buffer_width)

    async def _overlay_runtime_inputs(
        self,
        dut: Any,
        driver: Any,
        artifact: CompiledArtifact,
        symbol_table: dict[str, dict[str, Any]],
        tensor_names: list[str],
        values: dict[str, np.ndarray],
    ) -> None:
        for name in tensor_names:
            if name not in values:
                continue
            spec = artifact.plan.tensors[name]
            if spec.data is not None:
                continue
            symbol = symbol_table[name]
            packed = self._pack_tensor(values[name], symbol)
            for offset, word in enumerate(packed):
                await driver.write_reg(dut, driver.REG_ADDR, symbol["addr"] + offset, 16)
                await driver.write_reg(dut, driver.REG_CMD, 0x01, 8)
                await driver.write_reg(dut, driver.REG_MMVR, int(word), self.buffer_width)

    async def _load_im_image(self, dut: Any, driver: Any, instructions: list[int]) -> None:
        inst_width = 256
        num_chunks = max(1, inst_width // self.buffer_width)
        for index, inst in enumerate(instructions):
            for chunk_idx in range(num_chunks):
                chunk = (inst >> (chunk_idx * self.buffer_width)) & ((1 << self.buffer_width) - 1)
                addr = self.im_base_addr + (index * num_chunks) + chunk_idx
                await driver.write_reg(dut, driver.REG_ADDR, addr, 16)
                await driver.write_reg(dut, driver.REG_CMD, 0x01, 8)
                await driver.write_reg(dut, driver.REG_MMVR, chunk, self.buffer_width)

    async def _run_until_halt(self, dut: Any, driver: Any, RisingEdge: Any) -> None:
        doorbell_addr = driver.REG_MMVR + (self.buffer_width // 8) - 1
        await driver.write_reg(dut, driver.REG_ARG, self.im_base_addr, 32)
        await driver.write_reg(dut, driver.REG_CMD, 0x03, 8)
        await driver.write_reg(dut, doorbell_addr, 0, 8)
        for _ in range(100000):
            dut.host_addr.value = driver.REG_STATUS
            await RisingEdge(dut.clk)
            if int(dut.host_rd_data.value) == 0xFF:
                return
        raise AssertionError("Timeout waiting for HALT.")

    async def _read_tensor(self, dut: Any, driver: Any, spec: TensorSpec, symbol: dict[str, Any]) -> np.ndarray:
        role = symbol["role"]
        precision = PrecisionMode(symbol["precision"])
        if role != "C":
            raise NotImplementedError(
                f"Simulator readback currently supports only role C outputs, got role {role!r} for tensor {spec.name!r}."
            )
        return await self._read_role_c(dut, driver, spec.shape, symbol["addr"], precision)

    async def _capture_tensor_vectors(
        self,
        dut: Any,
        driver: Any,
        *,
        segment_name: str,
        symbol: dict[str, Any],
    ) -> dict[str, Any]:
        rows: list[list[int]] = []
        for offset in range(int(symbol["word_count"])):
            rows.append(await driver.read_ub_vector(dut, int(symbol["addr"]) + offset, self.array_size))
        return {
            "segment_name": segment_name,
            "addr": int(symbol["addr"]),
            "role": symbol["role"],
            "precision": PrecisionMode(symbol["precision"]).name,
            "rows": rows,
        }

    def _pack_tensor(self, value: np.ndarray, symbol: dict[str, Any]) -> list[int]:
        role = symbol["role"]
        precision = PrecisionMode(symbol["precision"])
        m_tiles, k_tiles, n_tiles = self._tile_counts(tuple(value.shape), role, precision)
        packed = self.packer.pack(np.array(value), role, precision, m_tiles, k_tiles, n_tiles)
        return [int(word) for word in packed]

    def _tile_counts(self, shape: tuple[int, int], role: str, precision: PrecisionMode) -> tuple[int, int, int]:
        p = 1 << (2 - precision)
        sz = self.array_size
        m = (shape[0] + sz - 1) // sz
        if role == "A":
            k = (shape[1] // p + sz - 1) // sz
            return m, k, 1
        if role == "B":
            k = (shape[0] // p + sz - 1) // sz
            n = (shape[1] + sz - 1) // sz
            return 1, k, n
        if role == "BIAS":
            return 1, 1, (shape[1] + sz - 1) // sz
        return m, 1, (shape[1] + sz - 1) // sz

    async def _read_role_c(self, dut: Any, driver: Any, shape: tuple[int, int], addr: int, precision: PrecisionMode) -> np.ndarray:
        actual = np.zeros(shape, dtype=np.int32)
        p = 1 << (2 - precision)
        bits = 16 // p
        mask = (1 << bits) - 1
        m_tiles = (shape[0] + self.array_size - 1) // self.array_size
        n_tiles = (shape[1] + self.array_size - 1) // self.array_size
        mt_phys = (m_tiles + p - 1) // p
        for mtp in range(mt_phys):
            for nt in range(n_tiles):
                tile_addr = addr + (mtp * n_tiles * self.array_size) + (nt * self.array_size)
                for row_in_tile in range(self.array_size):
                    vec = await driver.read_ub_vector(dut, tile_addr + row_in_tile, self.array_size)
                    for lane in range(self.array_size):
                        word = vec[lane]
                        col_idx = nt * self.array_size + lane
                        for bit_idx in range(p):
                            mt = mtp * p + bit_idx
                            row_idx = mt * self.array_size + row_in_tile
                            if row_idx < shape[0] and col_idx < shape[1]:
                                val = (word >> (bit_idx * bits)) & mask
                                if val & (1 << (bits - 1)):
                                    val -= (1 << bits)
                                actual[row_idx, col_idx] = val
        return actual


async def run_sim(
    artifact: CompiledArtifact,
    inputs: dict[str, np.ndarray],
    *,
    dut: Any,
    verification: VerificationMode = VerificationMode.OFF,
    reset: bool = False,
    defines_path: str | None = None,
    capture_vectors: bool = False,
):
    return await SimulatorExecutor(defines_path=defines_path).run(
        artifact,
        inputs,
        dut=dut,
        verification=verification,
        reset=reset,
        capture_vectors=capture_vectors,
    )
