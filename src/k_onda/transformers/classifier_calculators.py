from sklearn.cluster import KMeans as k_means
import numpy as np
import xarray as xr

from .core import Calculator


class KMeans(Calculator):
    require_all_finite = True
    accepted_data_types = (xr.DataArray,)

    def __init__(self, n_clusters=8, **kwargs):
        self._validate_configuration(n_clusters)
        self.n_clusters = n_clusters
        self.kmeans_kwargs = kwargs

    def _validate_configuration(self, n_clusters):
        if isinstance(n_clusters, bool) or not isinstance(n_clusters, int):
            raise TypeError(
                f"{self.format_call()}: n_clusters must be an integer."
            )
        if n_clusters < 1:
            raise ValueError(
                f"{self.format_call()}: n_clusters must be at least 1."
            )

    def _validate_data_schema(self, input_schema):
        super()._validate_data_schema(input_schema)

        expected_dims = {"index", "feature"}
        actual_dims = set(input_schema.dim_names)
        if actual_dims != expected_dims:
            raise ValueError(
                f"{self.format_call()}: input schema dimensions must be exactly "
                f"{sorted(expected_dims)!r}; received {sorted(actual_dims)!r}."
            )

    def _validate_data(self, data, **kwargs):
        units = data.pint.units
        values = np.asarray(data.pint.magnitude if units is not None else data)
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError(
                f"{self.format_call()}: feature values must be numeric."
            )

        super()._validate_data(data, **kwargs)

        if data.sizes["index"] < self.n_clusters:
            raise ValueError(
                f"{self.format_call()}: n_clusters={self.n_clusters} requires at "
                f"least {self.n_clusters} observations; received "
                f"{data.sizes['index']}."
            )

    def output_schema(self, input_schema):
        return input_schema.without("feature")

    def _apply_inner(self, data, *args, **kwargs):
        feature_matrix = data.transpose("index", "feature")
        units = feature_matrix.pint.units
        values = (
            feature_matrix.pint.magnitude
            if units is not None
            else feature_matrix.values
        )
        kmeans = k_means(n_clusters=self.n_clusters, **self.kmeans_kwargs)
        kmeans.fit(values)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        return labels, {"centers": centers}

    def _wrap_result(self, labels, data, centers=None):
        result = xr.DataArray(labels, dims=["index"])

        if "index" in data.coords:
            result = result.assign_coords({"index": data.coords["index"]})

        result = result.assign_attrs(
            data.attrs
            | {
                "kmeans_centers": centers,  # numpy array, shape (n_clusters, n_features)
                "kmeans_feature_names": list(data.coords["feature"].values),
            }
        )
        result = super()._wrap_result(result)
        return result
