import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tinynpu_jit import BenchmarkReport, PrimitiveCounts
from tinynpu_jit.inspect import format_benchmark_report


def test_end_to_end_speedup_includes_host_intrinsic_cycles():
    report = BenchmarkReport()
    report.add_entry(step="npu_seg", bucket="cpu_replaced", counts=PrimitiveCounts(adds=100))
    report.add_entry(step="npu_compute", bucket="npu_compute", cycles=10)
    report.add_entry(step="npu_overhead", bucket="npu_overhead", cycles=10)
    report.add_entry(step="host_softmax", bucket="host_intrinsic", cycles=20)

    assert report.pure_acceleration_speedup == 10.0
    assert report.integration_adjusted_speedup == 5.0
    assert report.end_to_end_speedup == 2.5

    payload = report.to_dict()
    assert payload["totals"]["end_to_end_speedup"] == 2.5


def test_format_benchmark_report_prints_end_to_end_speedup():
    report = BenchmarkReport()
    report.add_entry(step="cpu_ref", bucket="cpu_replaced", counts=PrimitiveCounts(adds=20))
    report.add_entry(step="npu_compute", bucket="npu_compute", cycles=5)
    report.add_entry(step="npu_overhead", bucket="npu_overhead", cycles=5)
    report.add_entry(step="host_softmax", bucket="host_intrinsic", cycles=10)

    class Result:
        benchmark = report

    formatted = format_benchmark_report(Result())
    assert "end_to_end_speedup: 1.0" in formatted
