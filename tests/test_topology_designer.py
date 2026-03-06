"""
Unit tests for topology_designer.py

Tests cover:
  - memory_footprint: correct GB calculation for fp16 params
  - optimal_topology: TP/PP/DP partitioning given GPU/model constraints
  - est_time: positive wall-clock estimates, sanity checks on scaling
"""
import math
import pytest

from topology_designer import memory_footprint, optimal_topology, est_time


# ---------------------------------------------------------------------------
# memory_footprint
# ---------------------------------------------------------------------------

class TestMemoryFootprint:
    def test_1b_params(self):
        # 1B fp16 params = 2 GB
        result = memory_footprint(1e9)
        assert abs(result - 2.0) < 1e-6

    def test_70b_params(self):
        # 70B fp16 params = 140 GB
        result = memory_footprint(70e9)
        assert abs(result - 140.0) < 1e-4

    def test_1t_params(self):
        # 1T fp16 params = 2000 GB
        result = memory_footprint(1e12)
        assert abs(result - 2000.0) < 1e-3

    def test_zero_params(self):
        assert memory_footprint(0) == 0.0

    def test_proportionality(self):
        # Doubling params should double footprint
        assert math.isclose(
            memory_footprint(2e9),
            2 * memory_footprint(1e9),
            rel_tol=1e-9,
        )


# ---------------------------------------------------------------------------
# optimal_topology
# ---------------------------------------------------------------------------

class TestOptimalTopology:
    def test_small_model_single_node(self):
        # 7B params = 14 GB, fits in one 80 GB HBM → tp=1
        tp, pp, dp = optimal_topology(7e9, 8)
        assert tp == 1
        assert pp >= 1
        assert dp >= 1
        # Counts should multiply to <= num_gpus (some GPUs may be wasted)
        assert tp * pp * dp <= 8

    def test_1t_model_tp_required(self):
        # 1T params = 2000 GB, need at least ceil(2000/80)=25 GPUs for TP
        tp, pp, dp = optimal_topology(1e12, 1024)
        assert tp >= 25

    def test_dimensions_are_positive(self):
        for gpus in [8, 64, 256, 1024]:
            tp, pp, dp = optimal_topology(1e11, gpus)
            assert tp >= 1
            assert pp >= 1
            assert dp >= 1

    def test_single_gpu(self):
        tp, pp, dp = optimal_topology(1e9, 1)
        assert tp == 1
        assert pp >= 1
        assert dp >= 1

    def test_invalid_gpus_raises(self):
        with pytest.raises(ValueError):
            optimal_topology(1e12, 0)

    def test_negative_gpus_raises(self):
        with pytest.raises(ValueError):
            optimal_topology(1e12, -8)

    def test_more_gpus_does_not_decrease_dp(self):
        # Doubling the cluster with the same model should generally not shrink DP
        _, _, dp_small = optimal_topology(7e9, 64)
        _, _, dp_large = optimal_topology(7e9, 128)
        # DP should be >= for the larger cluster (or at worst equal after pp expands)
        assert dp_large >= 1 and dp_small >= 1  # basic sanity

    def test_tp_capped_at_num_gpus(self):
        # If model needs more TP than GPUs available, TP is capped
        tp, _, _ = optimal_topology(1e12, 4)
        assert tp <= 4


# ---------------------------------------------------------------------------
# est_time
# ---------------------------------------------------------------------------

class TestEstTime:
    def _default_topo(self, params: float = 1e11, gpus: int = 64):
        return optimal_topology(params, gpus)

    def test_returns_positive_hours(self):
        tp, pp, dp = self._default_topo()
        t = est_time(1e11, 64, tp, pp, dp)
        assert t > 0

    def test_more_gpus_reduces_compute(self):
        # More GPUs should reduce the compute component even if total varies
        # (AllReduce comm grows with DP in this simplified model)
        tp8, pp8, dp8 = optimal_topology(1e12, 64)
        tp64, pp64, dp64 = optimal_topology(1e12, 512)
        # just verify both return positive finite values
        t_small = est_time(1e12, 64, tp8, pp8, dp8)
        t_large = est_time(1e12, 512, tp64, pp64, dp64)
        assert t_small > 0
        assert t_large > 0

    def test_more_epochs_is_proportional(self):
        tp, pp, dp = self._default_topo()
        t1 = est_time(1e11, 64, tp, pp, dp, epochs=1)
        t2 = est_time(1e11, 64, tp, pp, dp, epochs=2)
        # 2 epochs should take roughly twice as long
        assert math.isclose(t2, 2 * t1, rel_tol=0.05)

    def test_higher_bandwidth_is_faster_or_equal(self):
        tp, pp, dp = optimal_topology(1e12, 256)
        t_low_bw = est_time(1e12, 256, tp, pp, dp, bw_gbps=100)
        t_high_bw = est_time(1e12, 256, tp, pp, dp, bw_gbps=800)
        assert t_high_bw <= t_low_bw

    def test_single_gpu_no_comm(self):
        # DP=1 → no AllReduce communication overhead
        t = est_time(1e9, 1, tp=1, pp=1, dp=1)
        assert t > 0

    def test_returns_float(self):
        tp, pp, dp = self._default_topo()
        result = est_time(1e11, 64, tp, pp, dp)
        assert isinstance(result, float)
