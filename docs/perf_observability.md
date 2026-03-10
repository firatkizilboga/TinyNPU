## Optional Perf Observability

This NPU includes an optional performance/debug instrumentation layer for the control FSM.

### Design Goals

- No compiler dependency.
- No functional behavior change.
- Compile-time removable.
- Full control-state visibility for cycle analysis.

### What It Tracks

When `PERF_ENABLE=1`, the control unit exposes:

- `perf_total_cycles`: total cycles since reset
- `perf_state_cycles_flat`: 64-bit cycle counter for every control FSM state
- `perf_state_entries_flat`: 64-bit entry counter for every control FSM state
- `perf_state_id`: current FSM state ID

Tracked states:

- `CTRL_IDLE`
- `CTRL_HOST_WRITE`
- `CTRL_HOST_READ`
- `CTRL_READ_WAIT`
- `CTRL_FETCH`
- `CTRL_DECODE`
- `CTRL_EXEC_MOVE`
- `CTRL_EXEC_MATMUL`
- `CTRL_MM_CLEAR`
- `CTRL_MM_FEED`
- `CTRL_MM_WAIT`
- `CTRL_MM_LOAD_BIAS`
- `CTRL_MM_DRAIN_SA`
- `CTRL_MM_WRITEBACK`
- `CTRL_HALT`

### Build Modes

- Default build: `PERF_ENABLE=0`
  - counters stay inactive
  - normal execution path remains unchanged
- Perf/debug build: `PERF_ENABLE=1`
  - counters are enabled for cocotb or waveform inspection

### Running The Perf Test

From `verification/cocotb`:

```sh
CCACHE_DISABLE=1 USER_EXTRA_ARGS='-GPERF_ENABLE=1' MODULE=test_perf_observability make -f Makefile.npu
```

Optional workload override:

```sh
CCACHE_DISABLE=1 USER_EXTRA_ARGS='-GPERF_ENABLE=1' MODULE=test_perf_observability NPU_FILE=../../software/workload/simple_chain.npu make -f Makefile.npu
```

### How To Use The Counters

For phase-level analysis, snapshot counters before and after the region of interest and take a delta.

Example:

- snapshot after program load
- snapshot after `HALT`
- subtract to get execution-only state costs

This is intended as a debug/perf tool, not a required production interface.
