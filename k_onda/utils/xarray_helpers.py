import numpy as np


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


def standardize(num_array):
    return np.round(num_array, 8).astype(np.float64)

