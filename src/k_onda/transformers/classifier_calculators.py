from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Real

from sklearn.cluster import KMeans as k_means
import numpy as np
import xarray as xr

from .core import Calculator
from k_onda.utils import ValidationMixin


@dataclass(frozen=True)
class SklearnKMeans(ValidationMixin):
    init: (
        str | Callable[..., object] | np.ndarray | Sequence[Sequence[Real]]
    ) = "k-means++"
    n_init: int | str = "auto"
    max_iter: int = 300
    tol: Real = 0.0001
    verbose: int = 0
    random_state: int | np.random.RandomState | None = None
    copy_x: bool = True
    algorithm: str = "lloyd"

    def __post_init__(self):
        self.validate_type_hints()
        if isinstance(self.init, str):
            self.validate_parameter(
                "init", self.init, choices=("k-means++", "random")
            )

        if isinstance(self.n_init, str):
            self.validate_parameter("n_init", self.n_init, choices=("auto",))
        else:
            self.validate_number(
                "n_init", self.n_init, number_type=int, minimum=1
            )

        self.validate_number(
            "max_iter", self.max_iter, number_type=int, minimum=1
        )
        self.validate_number("tol", self.tol, minimum=0, finite=True)
        self.validate_number(
            "verbose", self.verbose, number_type=int, minimum=0
        )
        if isinstance(self.random_state, bool):
            raise TypeError(
                f"{self.format_call()}: random_state must not be a boolean."
            )
        self.validate_parameter(
            "algorithm", self.algorithm, choices=("lloyd", "elkan")
        )

        if isinstance(self.init, Sequence) and not isinstance(
            self.init, (str, np.ndarray)
        ):
            if not self.init:
                raise ValueError(f"{self.format_call()}: init cannot be empty.")
        if isinstance(self.init, np.ndarray) and self.init.size == 0:
            raise ValueError(f"{self.format_call()}: init cannot be empty.")

    def __call__(self, values, n_clusters):
        estimator = k_means(
            n_clusters=n_clusters,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=self.verbose,
            random_state=self.random_state,
            copy_x=self.copy_x,
            algorithm=self.algorithm,
        )
        estimator.fit(values)
        return estimator.labels_, estimator.cluster_centers_


class KMeans(Calculator):
    require_all_finite = True
    accepted_data_types = (xr.DataArray,)

    def __init__(
        self,
        n_clusters: int = 8,
        *,
        implementation: Callable[..., tuple[object, object]] | None = None,
    ):
        self.validate_type_hints()
        self.validate_number(
            "n_clusters", n_clusters, number_type=int, minimum=1
        )
        self.n_clusters = n_clusters
        self.implementation = (
            SklearnKMeans() if implementation is None else implementation
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
        labels, centers = self.implementation(values, self.n_clusters)
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
