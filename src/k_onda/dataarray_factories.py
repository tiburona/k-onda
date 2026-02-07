
import numpy as np
import xarray as xr

from .central import ureg


def get_time_coords(data, start, sampling_rate=None, dt=None):
    if sampling_rate is None and dt is None:
        raise ValueError("One of sampling_rate and dt must have a value.")
    dt = dt or (1 / sampling_rate)
    dt = getattr(dt, "magnitude", dt)
    start = getattr(start, "magnitude", start)
    time = (np.arange(len(data)) * dt) + start
    return time


def _ensure_attrs(attrs, **known_attrs):
    attrs = attrs or {}
    attrs.update(known_attrs)
    return attrs


def make_time_series(data, sampling_rate, start = 0 * ureg.s, units="s", data_units=None, attrs=None):
    
    time = get_time_coords(data, start, sampling_rate=sampling_rate)
        
    attrs = _ensure_attrs(getattr(data, 'attrs', {}), sampling_rate=sampling_rate)

    return make_data_series(
        data, 
        dims=["time"], 
        units=[units], 
        coords=[time], 
        data_units=data_units, 
        attrs=attrs)


def make_time_frequency_series(
    data,
    freq,
    start = 0 * ureg.s,
    dt = None,
    sampling_rate = None,
    freq_units="Hz",  
    time_units="s", 
    data_units=None, 
    attrs=None):

    # pass the first col
    time = get_time_coords(data[0, :], start=start, sampling_rate=sampling_rate, dt=dt)

    attrs = _ensure_attrs(getattr(data, 'attrs', {}), sampling_rate=sampling_rate)
    
    return make_data_series(
        data, 
        dims=["frequency", "time"],
        units=[freq_units, time_units],
        coords=[freq, time],
        data_units=data_units,
        attrs=attrs)


def make_data_series(data, dims, units, coords, data_units=None, attrs=None):
    
    da = xr.DataArray(data, dims=dims, attrs=attrs or {})

    da = da.assign_coords({
        dim: getattr(coord, "magnitude", coord) 
        for dim, coord in zip(dims, coords)
    })

    da = da.pint.quantify({
        dim: unit for dim, unit in zip(dims, units)
    })

    if data_units is not None:
        da = da.pint.quantify(data_units)

    return da
   

    

    

