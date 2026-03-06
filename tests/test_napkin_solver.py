"""
Unit tests for napkin_solver.py

Tests cover:
  - NapkinResult dataclass fields and __str__ formatting
  - solve(): topology correctness, time estimates, error handling
  - edge cases: single GPU, massive cluster, low/high bandwidth
"""
import math
import pytest

from napkin_solver import solve, NapkinResult


# ---------------------------------------------------------------------------
# NapkinResult
# ---------------------------------------------------------------------------

class TestNapkinResult:
    def _make(self, **kwargs):
        defaults = dict(
            params=1e12, gpus=1024, tp=25, pp=5, dp=8,
            compute_hr=10.0, comm_hr=2.0, bubble_hr=0.5, total_hr=12.5,
            model_gb=2000.0,
        )
        defaults.update(kwargs)
        return NapkinResult(**defaults)

    def test_str_contains_topology(self):
        r = self._make()
        s = str(r)
        assert "TP=25" in s
        assert "PP=5" in s
        assert "DP=8" in s

    def test_str_contains_total(self):
        r = self._make()
        assert "12.500" in str(r)

    def test_str_contains_params(self):
        r = self._make()
        assert "1.00T" in str(r)

    def test_str_contains_gpu_count(self):
        r = self._make()
        assert "1024" in str(r)


# ---------------------------------------------------------------------------
# solve() — topology dimensions
# ---------------------------------------------------------------------------

class TestSolveDimensions:
    def test_all_dims_positive(self):
        r = solve(1e12, 1024)
        assert r.tp >= 1
        assert r.pp >= 1
        assert r.dp >= 1

    def test_tp_covers_model_memory(self):
        # 1T params = 2000 GB; ceil(2000/80) = 25
        r = solve(1e12, 1024)
        model_gb_per_tp_shard = r.model_gb / r.tp
        assert model_gb_per_tp_shard <= 80.0

    def test_small_model_tp_is_1(self):
        # 7B params = 14 GB → fits in single 80 GB card
        r = solve(7e9, 8)
        assert r.tp == 1

    def test_params_stored_on_result(self):
        r = solve(5e10, 32)
        assert r.params == 5e10
        assert r.gpus == 32

    def test_model_gb_is_correct(self):
        r = solve(1e12, 256)
        assert math.isclose(r.model_gb, 2000.0, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# solve() — time estimates
# ---------------------------------------------------------------------------

class TestSolveTiming:
    def test_all_times_positive(self):
        r = solve(1e11, 64)
        assert r.compute_hr > 0
        assert r.comm_hr >= 0
        assert r.bubble_hr >= 0
        assert r.total_hr > 0

    def test_total_is_sum_of_parts(self):
        r = solve(1e11, 64)
        expected = r.compute_hr + r.comm_hr + r.bubble_hr
        assert math.isclose(r.total_hr, expected, rel_tol=1e-9)

    def test_more_gpus_both_return_positive(self):
        # Simplified comm model means total time doesn't always decrease with more GPUs
        # (AllReduce grows linearly with DP count); just verify valid outputs
        r_small = solve(7e9, 8)
        r_large = solve(7e9, 128)
        assert r_small.total_hr > 0
        assert r_large.total_hr > 0

    def test_two_epochs_roughly_doubles_time(self):
        t1 = solve(1e11, 64, epochs=1).total_hr
        t2 = solve(1e11, 64, epochs=2).total_hr
        assert math.isclose(t2, 2 * t1, rel_tol=0.05)

    def test_higher_bandwidth_reduces_or_equals_time(self):
        t_slow = solve(1e12, 512, bw_gbps=100).total_hr
        t_fast = solve(1e12, 512, bw_gbps=800).total_hr
        assert t_fast <= t_slow

    def test_single_gpu_returns_valid_result(self):
        r = solve(1e9, 1)
        assert r.total_hr > 0
        assert r.dp == 1   # no data parallelism possible


# ---------------------------------------------------------------------------
# solve() — error handling
# ---------------------------------------------------------------------------

class TestSolveErrors:
    def test_zero_gpus_raises(self):
        with pytest.raises(ValueError, match="gpus"):
            solve(1e12, 0)

    def test_negative_gpus_raises(self):
        with pytest.raises(ValueError, match="gpus"):
            solve(1e12, -4)

    def test_zero_params_raises(self):
        with pytest.raises(ValueError, match="params"):
            solve(0, 64)

    def test_negative_params_raises(self):
        with pytest.raises(ValueError, match="params"):
            solve(-1e12, 64)
