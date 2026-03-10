CTRL_STATE_NAMES = [
    "CTRL_IDLE",
    "CTRL_HOST_WRITE",
    "CTRL_HOST_READ",
    "CTRL_READ_WAIT",
    "CTRL_FETCH",
    "CTRL_DECODE",
    "CTRL_EXEC_MOVE",
    "CTRL_EXEC_MATMUL",
    "CTRL_MM_CLEAR",
    "CTRL_MM_FEED",
    "CTRL_MM_WAIT",
    "CTRL_MM_LOAD_BIAS",
    "CTRL_MM_DRAIN_SA",
    "CTRL_MM_WRITEBACK",
    "CTRL_HALT",
]

PERF_COUNTER_WIDTH = 64


def unpack_flat_counters(flat_value):
    counters = {}
    mask = (1 << PERF_COUNTER_WIDTH) - 1
    for idx, name in enumerate(CTRL_STATE_NAMES):
        counters[name] = (int(flat_value) >> (idx * PERF_COUNTER_WIDTH)) & mask
    return counters


def diff_counters(after, before):
    return {name: after[name] - before[name] for name in CTRL_STATE_NAMES}
