import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import inference_benchmark as benchmark


class InferenceBenchmarkTests(unittest.TestCase):
    def test_summarize_latencies_reports_median_mean_p95_and_bounds(self):
        result = benchmark.summarize_latencies_ms([1.0, 2.0, 3.0, 4.0])

        self.assertEqual(result["count"], 4)
        self.assertAlmostEqual(result["median_ms"], 2.5)
        self.assertAlmostEqual(result["mean_ms"], 2.5)
        self.assertAlmostEqual(result["p95_ms"], 3.85)
        self.assertAlmostEqual(result["min_ms"], 1.0)
        self.assertAlmostEqual(result["max_ms"], 4.0)

    def test_benchmark_forward_runs_warmups_outside_recorded_samples(self):
        calls = []
        ticks = iter([0.0, 0.01, 1.0, 1.02, 2.0, 2.03])

        with patch.object(benchmark.time, "perf_counter", side_effect=lambda: next(ticks)):
            result = benchmark.benchmark_forward(
                lambda: calls.append("forward"), warmup_runs=1, timed_runs=2
            )

        self.assertEqual(calls, ["forward", "forward", "forward"])
        self.assertAlmostEqual(result[0], 10.0)
        self.assertAlmostEqual(result[1], 20.0)
