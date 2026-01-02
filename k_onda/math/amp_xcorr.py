import numpy as np
import xarray as xr

def fisher_z_from_r(r_da: xr.DataArray, eps=1e-12) -> xr.DataArray:
    r = r_da.clip(min=-1 + eps, max=1 - eps)
    return xr.apply_ufunc(np.arctanh, r, keep_attrs=True)


def back_transform_fisher_z(z_da: xr.DataArray) -> xr.DataArray:
    return xr.apply_ufunc(np.tanh, z_da, keep_attrs=True) 
