import numpy as np
import xarray as xr

from k_onda.utils import is_truthy



def drop_inconsistent_coords(arrs, tol=1e-8):
    """
    For each common coordinate, keep it only if *all* arrays agree (within tol).
    If they don’t, wipe it out—but
      • for ordinary coords → set to None
      • for coords that are also dimensions → replace with 0…N-1 index
    """
    cleaned = list(arrs)
    coord_names = set.intersection(*(set(a.coords) for a in arrs))

    for name in coord_names:
        ref_vals = arrs[0].coords[name].values

        # --- skip if this coord is already 'None' in any array ----------
        if _is_none_coord(ref_vals):
            continue
        if any(_is_none_coord(a.coords[name].values) for a in arrs[1:]):
            continue


        # --- if values differ, normalise them ---------------------------
        if not all(name in a.coords and
                   np.allclose(a.coords[name].values, ref_vals, atol=tol)
                   for a in arrs[1:]):

            for i, a in enumerate(cleaned):
                if name in a.coords:
                    cleaned[i] = a.drop_vars(name)

                if name in a.dims:
                    # dimension coord → recreate as simple index
                    n = a.sizes[name]
                    cleaned[i] = cleaned[i].assign_coords({name: np.arange(n)})
                else:
                    # ordinary coord → blank it out
                    cleaned[i] = cleaned[i].assign_coords({name: None})

    return tuple(cleaned)


def _is_none_coord(vals):
    """True if the coord is the scalar object() placeholder xarray uses for None."""
    return (
        hasattr(vals, "shape") and vals.shape == ()
        and vals.dtype == object and vals.item() is None
    )


def round_coords(arrs, *, decimals=8):
    """
    Return a list of DataArrays where **every numeric coordinate common to
    all arrays** is rounded to `decimals` and made byte-identical.

    Works with *any* coordinate names; no list required.
    """
    if len(arrs) <= 1:
        return list(arrs)

    # coords present in every array
    common = set.intersection(*(set(a.coords) for a in arrs))

    # build rounded refs from the first array
    ref = arrs[0]
    rounded_refs = {}
    for name in common:
        vals = ref.coords[name].values
        if np.issubdtype(vals.dtype, np.number):
            rounded_vals = np.round(vals, decimals=decimals)
            dims = ref.coords[name].dims        # ('time',) or ('y','x'), etc.
            rounded_refs[name] = (dims, rounded_vals)

    # stamp each array with the rounded refs
    out = []
    for a in arrs:
        a2 = a
        for name, (dims, vals) in rounded_refs.items():
            if name in a2.coords:
                a2 = a2.assign_coords({name: (dims, vals)})
        out.append(a2)

    return out


def _nan_like_dataarray(da):
    """Return a DataArray with the same shape/coords but filled with NaNs."""
    return xr.full_like(da, np.nan)


def _nan_like_dataset(ds):
    """Return a Dataset with every data var filled with NaNs."""
    data_vars_nan = {
        k: xr.full_like(v, np.nan) for k, v in ds.data_vars.items()
    }
    # coords & attrs are copied automatically when you build a new Dataset
    return xr.Dataset(
        data_vars=data_vars_nan,
        coords={k: v.copy() for k, v in ds.coords.items()},
        attrs=ds.attrs,
    )


def fill_missing_arrays(seq):
    """
    Replace scalar NaNs/None in *seq* with xr objects that match the first
    non-NaN exemplar in structure, shape and dtype.
    """
    exemplar = next(
        (o for o in seq if isinstance(o, (xr.DataArray, xr.Dataset))),
        None,
    )
    if exemplar is None:
        raise ValueError("No xr object found to infer shape from.")

    if isinstance(exemplar, xr.DataArray):
        make_nan = lambda: _nan_like_dataarray(exemplar)
    else:  # exemplar is Dataset
        make_nan = lambda: _nan_like_dataset(exemplar)

    filled = []
    for o in seq:
        if isinstance(o, (xr.DataArray, xr.Dataset)):
            filled.append(o)
        elif o is None or (isinstance(o, float) and np.isnan(o)):
            filled.append(make_nan())
        else:
            raise TypeError(
                f"Unsupported type {type(o)}: expected xr object, None, or NaN."
            )
    return filled


class XMean:

    @staticmethod
    def clean_and_aggregate(child_vals):
        cleaned = drop_inconsistent_coords(child_vals)
        cleaned = round_coords(cleaned, decimals=8)
        agg = xr.concat(cleaned, dim="child", coords="minimal", compat="no_conflicts")
        return agg
    
    @staticmethod
    def _as_da_scalar(x):
        if isinstance(x, xr.DataArray):
            return x.squeeze(drop=True)
        return xr.DataArray(np.asarray(x).squeeze())
    
    @staticmethod
    # ---- helper: mean or weighted mean over a named dim ----
    def _maybe_weighted_mean(obj, dim, weights):
        def _apply_mean(da: xr.DataArray):
            if weights is None:
                return da.mean(dim=dim, skipna=True, keep_attrs=True)

            w = weights
            if not isinstance(w, xr.DataArray):
                w = xr.DataArray(np.asarray(w))

            if dim is not None:
                if w.ndim == 1 and (len(w.dims) == 0 or w.dims[0] != dim):
                    w = xr.DataArray(w.data, dims=(dim,))
                if dim in da.dims and dim in w.dims:
                    if da.sizes[dim] != w.sizes[dim]:
                        raise ValueError(f"weights length along '{dim}' ({w.sizes[dim]}) "
                                            f"does not match data ({da.sizes[dim]}).")
                    w_aligned = w.assign_coords({dim: da[dim]})
                else:
                    w_aligned = w

                w_aligned = w_aligned / w_aligned.sum(dim=dim)
                da_aligned, w_aligned = xr.align(da, w_aligned, join="exact", copy=False)
                return (da_aligned * w_aligned).sum(dim=dim, skipna=True, keep_attrs=True)

            w_norm = w / w.sum()
            da_b, w_b = xr.broadcast(da, w_norm)
            return (da_b * w_b).sum(skipna=True, keep_attrs=True)

        if isinstance(obj, xr.Dataset):
            data_vars = {k: _apply_mean(v) for k, v in obj.data_vars.items()}
            result = xr.Dataset(data_vars)
            # retain shared coords/attrs when present
            return result.assign_coords(obj.coords).assign_attrs(obj.attrs)
        return _apply_mean(obj)


    def xmean(self, child_vals, axis=None, weights=None):
        """
        Always returns an xr.DataArray.

        - If child_vals is empty/falsey (but zeros are allowed), return a NaN scalar DataArray.
        - If child_vals is a scalar or size-1 array/DataArray, return a *scalar* (0-D) DataArray.
        - Otherwise, compute the (optionally weighted) mean (optionally along `axis`), preserving attrs.
        """
        if not is_truthy(child_vals, zero_ok=True):
            return xr.DataArray(np.nan)

        if isinstance(child_vals, (float, np.floating)) or \
        (isinstance(child_vals, (np.ndarray, xr.DataArray)) and np.size(child_vals) == 1):
            return self._as_da_scalar(child_vals)

        # ---- axis handling ----
        if axis is None:
            if isinstance(child_vals, xr.DataArray):
                return self._maybe_weighted_mean(child_vals, dim=None, weights=weights)

            agg = self.clean_and_aggregate(child_vals)
            return self._maybe_weighted_mean(agg, dim="child")

        # axis provided
        if isinstance(child_vals, xr.DataArray):
            axis_name = child_vals.dims[axis] if isinstance(axis, int) else axis
            return self._maybe_weighted_mean(child_vals, dim=axis_name, weights=weights)
        
        agg = self.clean_and_aggregate(child_vals)
        
        if isinstance(axis, int):
            if isinstance(agg, xr.Dataset):
                dim_names = list(agg.dims)
                axis_name = dim_names[axis]
            else:
                axis_name = agg.dims[axis]
        else:
            axis_name = axis or "child"
        return self._maybe_weighted_mean(agg, dim=axis_name, weights=weights)