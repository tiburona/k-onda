
import numpy as np
import xarray as xr




def get_time_coords(data, sampling_rate):
    dt = (1 / sampling_rate).to("s")
    time = np.arange(len(data)) * dt.magnitude
    return time


def make_time_series_from_fs(data, sampling_rate, units="s", data_units=None, attrs=None):
    time = get_time_coords(data, sampling_rate)
    return make_time_series(data, time, units, data_units, attrs)


def make_time_series(data, time, units="s", data_units=None, attrs=None):

    return make_data_series(
        data, 
        dims=["time"], 
        units=[units], 
        coords=[time], 
        data_units=data_units, 
        attrs=attrs)


def make_frequency_time_series(
        data, 
        freq,
        time,
        freq_units="Hz",  
        time_units="s", 
        data_units=None, 
        attrs=None):
    
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
   

    

    

