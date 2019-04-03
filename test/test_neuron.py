import numpy as np
from spyky.neuron import IFNeuron


class TestIFNeuron:
    def test_tick(self):
        v_rest = -70
        refractor_period = 3
        n = IFNeuron(
            shape=(2, 2), v_rest=v_rest, refractor_period=refractor_period
        )
        n.dt = 1  # needed because we have no network

        inpts = np.array([[1, 2], [2, 3]])
        n.tick(inpts)
        v_expect = v_rest + inpts
        assert all([a == b for a, b in zip(n.v.flatten(), v_expect.flatten())])
        assert all(n.spikes.flatten() == 0)
        assert all(n.refractor_count.flatten() == 0)

    def test_tick_with_mask_clipping(self):
        v_rest = -70
        v_thresh = 20
        refractor_period = 3
        n = IFNeuron(
            shape=(2, 2),
            v_rest=v_rest,
            v_thresh=v_thresh,
            refractor_period=refractor_period,
        )
        n.dt = 1  # needed because we have no network

        inpts = np.array([[100, 100], [100, 100]])  # over v_thresh
        n.tick(inpts)
        check_clipping_upper = n.v > v_thresh
        check_clipping_lower = n.v < v_rest
        assert all(n.refractor_count.flatten() == refractor_period)
        assert all(n.spikes.flatten() == 1)
        assert all(n.v.flatten() == v_rest)
        assert all(check_clipping_upper.flatten() == 0)
        assert all(check_clipping_lower.flatten() == 0)

    def test_reset(self):
        v_rest = -50
        n = IFNeuron(shape=(2, 2), v_rest=v_rest)
        v_before_reset = n.v
        spikes_before_reset = n.spikes
        refractor_count_before_reset = n.refractor_count
        n.reset()
        assert all(v_before_reset.flatten() == v_rest)
        assert all(n.v.flatten() == v_rest)
        assert all(spikes_before_reset.flatten() == 0)
        assert all(n.spikes.flatten() == 0)
        assert all(refractor_count_before_reset.flatten() == 0)
        assert all(n.refractor_count.flatten() == 0)
