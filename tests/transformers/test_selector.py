import numpy as np
import xarray as xr

from k_onda.central import DimBounds, DimPair
from k_onda.transformers.selector.selector import Slicer


class FakeLocus:
    dim = 'time'
    metadim = 'time'


def test_continuous_relative_coords_start_at_window_start():
    data = xr.DataArray(
        np.arange(3),
        dims=('time',),
        coords={'time': np.array([10.0, 11.0, 12.0])},
    )
    window = DimBounds({'time': DimPair([-0.05, 0.3])})
    slicer = Slicer(None, 'local', FakeLocus(), 'pip', window)

    result = slicer.attach_continuous_relative_coords([data])[0]

    expected = np.array([-0.05, 0.95, 1.95])
    np.testing.assert_allclose(result.coords['relative_time'].data, expected)
    np.testing.assert_allclose(result.coords['pip_time'].data, expected)
