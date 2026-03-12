from sklearn.cluster import KMeans as k_means
import xarray as xr

from .core import Calculator, Transformer

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

        result = result.assign_attrs(
            data.attrs | {
                'kmeans_centers': centers,  # numpy array, shape (n_clusters, n_features)
                'kmeans_feature_names': list(data.coords['feature'].values)
            })

        return result


class ApplyLabels(Transformer):

    def __init__(self, label_name, label_list=None, label_func=None):
        self.label_name = label_name
        self.label_list = label_list
        self.label_func = label_func
        if self.label_list is None and self.label_func is None:
            raise ValueError("One of label_list and label_func must not be None.")

    def _validate_input(self, input):
        from k_onda.signals import IndexedSignal
        if not isinstance(input, IndexedSignal):
            raise ValueError("Input must be of type IndexedSignal to apply labels.")
    
    def _get_transform(self, ):
        pass

    def _apply(self, data):
        entities_to_label = data.coords['index'].values
        if self.label_list:
            labels = {i:lab for i, lab in enumerate(self.label_list)}
        else:
            # example label_func

            def label_func(centers, feature_names):
                pass

            kmeans_feature_names = data.attrs('kmeans_feature_names')
            centers = data.attrs('kmeans_centers')  # numpy array, shape (n_clusters, n_features)
            normalization_params = data.attrs.get('normalization_params')
            feature_units = data.attrs('feature_units')
            labels = self.label_func(centers, kmeans_feature_names)

    def _denormalize_centers(centers, normalization_params):
        if 'rms' in normalization_params:
            return 

