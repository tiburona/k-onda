import numpy as np

def drop_inconsistent_coords(arrs, tol=1e-8):
    cleaned = list(arrs)
    coord_names = set.intersection(*(set(a.coords) for a in arrs))
    for name in coord_names:
        ref_vals = arrs[0].coords[name].values
        # skip coords already dropped (represented as array(None))
        if hasattr(ref_vals, "shape") and ref_vals.shape == () and ref_vals.dtype == object and ref_vals.item() is None:
            continue
        # also skip if any other array's coord was previously dropped
        skip = False
        for a in arrs[1:]:
            other = a.coords[name].values
            if hasattr(other, "shape") and other.shape == () and other.dtype == object and other.item() is None:
                skip = True
                break
        if skip:
            continue
        # drop if values mismatch
        if not all(
            name in a.coords and np.allclose(a.coords[name].values, ref_vals, atol=tol)
            for a in arrs[1:]
        ):
            for i in range(len(cleaned)):
                cleaned[i] = cleaned[i].assign_coords({name: None})
    return tuple(cleaned)


def standardize(num_array):
        return np.round(num_array, 8).astype(np.float64)

