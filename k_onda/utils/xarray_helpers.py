import numpy as np

def drop_inconsistent_coords(arrs, tol=1e-8):
    coord_names = set.intersection(*(set(a.coords) for a in arrs))
    cleaned = list(arrs)

    for name in coord_names:
        if name in arrs[0].indexes:
            continue  # Don't drop index coords like 'time'

        ref_vals = arrs[0].coords[name].values
        if not all(
            name in a.coords and np.allclose(a.coords[name].values, ref_vals, atol=tol)
            for a in arrs[1:]
        ):
            for i in range(len(cleaned)):
                cleaned[i] = cleaned[i].assign_coords({name: None})

    return cleaned


def standardize(num_array):
        return np.round(num_array, 8).astype(np.float64)

