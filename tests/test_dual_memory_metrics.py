"""Dual metric: Lychee write cost + MemCon read cost."""

from benchmarks.eval_memory import run_dual_benchmark


def test_dual_benchmark_reports_write_and_read_savings():
    result = run_dual_benchmark()

    assert set(result) >= {"read", "write", "workload_turns"}

    read = result["read"]
    assert read["naive_tokens"] > read["memcon_tokens"]
    assert read["reduction_pct"] > 0

    write = result["write"]
    assert write["eager_extract_calls"] > write["lychee_extract_calls"]
    assert write["eager_tokens"] > write["lychee_tokens"]
    assert write["reduction_pct"] > 0
