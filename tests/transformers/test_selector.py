import pytest
import numpy as np
import xarray as xr

from k_onda.transformers import Window
from k_onda.central import ureg


@pytest.fixture
def spike_dataset():
    """Five spikes at 0.1, 0.5, 1.0, 1.5, 2.0 seconds.
    coord_map = {'time': 'spike_times'} mirrors what SpikeCluster sets.
    """
    times = np.array([0.1, 0.5, 1.0, 1.5, 2.0]) * ureg.s
    return xr.Dataset({"spike_times": xr.DataArray(times, dims=["spikes"])})


@pytest.fixture
def coord_map():
    return {"time": "spike_times"}


def _make_window(t0_s, t1_s):
    return Window(
        selection_endpoints={"time": [t0_s * ureg.s, t1_s * ureg.s]},
        selector_mode="pushdown",
    )


class TestGetPointsInterval:
    """Window.get_points uses a half-open interval [t0, t1)."""

    def test_interior_spikes_selected(self, spike_dataset, coord_map):
        win = _make_window(0.4, 1.5)
        point_dim, selected = win.get_points(spike_dataset, coord_map, {"time"})

        assert point_dim == "spikes"
        assert selected == {1, 2}  # 0.5 and 1.0 are in [0.4, 1.5)

    def test_spike_at_lower_bound_is_included(self, spike_dataset, coord_map):
        win = _make_window(0.5, 1.0)
        _, selected = win.get_points(spike_dataset, coord_map, {"time"})

        assert 1 in selected  # spike at 0.5 s

    def test_spike_at_upper_bound_is_excluded(self, spike_dataset, coord_map):
        win = _make_window(0.5, 1.0)
        _, selected = win.get_points(spike_dataset, coord_map, {"time"})

        assert 2 not in selected  # spike at 1.0 s

    def test_empty_window_returns_empty_set(self, spike_dataset, coord_map):
        win = _make_window(0.6, 0.9)
        _, selected = win.get_points(spike_dataset, coord_map, {"time"})

        assert selected == set()

    def test_no_map_dims_returns_all_indices(self, spike_dataset, coord_map):
        win = _make_window(0.0, 3.0)
        _, selected = win.get_points(spike_dataset, coord_map, set())

        assert selected == set(range(5))