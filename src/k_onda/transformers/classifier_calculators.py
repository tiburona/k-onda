from sklearn.cluster import KMeans as k_means
import xarray as xr

from .core import Calculator

class KMeans(Calculator):

    def __init__(self, n_clusters=8, **kwargs):
        self.n_clusters=n_clusters
        self.kmeans_kwargs=kwargs

    def _apply_inner(self, data):
        kmeans = k_means(n_clusters=self.n_clusters, **self.kmeans_kwargs)
        kmeans.fit(data)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        return labels, {'centers': centers}

    def _wrap_result(self, labels, data, centers=None):
        result = xr.DataArray(labels, dims=['index'])

        if 'index' in data.coords:
            result = result.assign_coords ({'index': data.coords['index']})

        result = result.assign_attrs({'kmeans_centers': centers})

        return result
