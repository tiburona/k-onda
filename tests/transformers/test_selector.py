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


@pytest.fixture
def frequency_dataset():
    freqs = np.array([4.0, 5.0, 6.0, 7.0]) * ureg.Hz
    times = np.array([0.1, 0.5, 1.0, 1.5, 2.0]) * ureg.s
    power = np.array([[20.0, 30.0, 30.0, 30.0, 20.0],
                      [40.0, 50.0, 50.0, 50.0, 40.0],
                      [50.0, 60.0, 60.0, 80.0, 60.0],
                      [20.0, 30.0, 30.0, 30.0, 20.0]])
    return xr.DataArray(
        power,
        dims=('time', 'freq'),
        coords={
            'time': times,
            'freqs': freqs
        }
    )


@pytest.fixture
def two_coord_map():
    return {"x": "x_pos", "y": "y_pos"}


def _make_two_coord_window(x0, x1, y0, y1, x_unit=ureg.mm, y_unit=ureg.mm):

    return Window(
        selection_endpoints={
            "x": [x0 * x_unit, x1 * x_unit],
            "y": [y0 * y_unit, y1 * y_unit]},
        selector_mode="pushdown",
    )


@pytest.fixture
def position_dataset():
    x_pos = np.array([0.1, 0.5, 1.0, 1.5, 2.0]) * ureg.mm 
    y_pos = np.array([0.3, 1.5, 3, 4.5, 6.0]) * ureg.mm 
    return xr.Dataset({
        "x_pos": xr.DataArray(x_pos, dims=["rears"]),
        "y_pos": xr.DataArray(y_pos, dims=["rears"])
        })


class TestGetPointsOnTwoCoordMap:

    def test_interior_postitions_selected(self, position_dataset, two_coord_map):
        win = _make_two_coord_window(0.4, 1.5, 1.4, 3.5)
        point_dim, selected = win.get_points(position_dataset, two_coord_map, {"x", "y"})

        assert point_dim == "rears"
        assert selected == {1, 2} 


class TestSelectPointProcess:
   
    def test_returns_dataset_with_correct_rows(self, spike_dataset, coord_map):
        window = _make_window(0.4, 1.5)
        result = window.select_point_process(spike_dataset, coord_map)
        expected_times = np.array([ 0.5, 1.0]) * ureg.s

        assert isinstance(result, xr.Dataset)
        assert list(result.keys()) == ['spike_times']
        assert result['spike_times'].equals(
            xr.DataArray(expected_times, dims=["spikes"])
            )



