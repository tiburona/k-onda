import numpy as np
import xarray as xr

def make_array(offset):
    time = np.arange(20)
    bin_size = 0.5
    start = offset
    abs_time = time * bin_size + start
    rel_time = abs_time - offset  # = time * bin_size
    per_time = rel_time  # all aligned to same

    return xr.DataArray(
        np.random.rand(20),
        dims=["time"],
        coords={
            "time": time,
            "absolute_time": ("time", abs_time),
            "relative_time": ("time", rel_time),
            "period_time": ("time", per_time),
        },
        name=None,
    )

arrs = [make_array(0.0)] + [make_array(0.001 * i) for i in range(1, 5)]

# drop inconsistent coords (simulate your pipeline)
for i in range(len(arrs)):
    arr = arrs[i]
    arr = arr.assign_coords(absolute_time=None)
    arrs[i] = arr

xr.concat(arrs, dim="child", coords="minimal", compat="no_conflicts")