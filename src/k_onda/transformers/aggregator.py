import xarray as xr

from k_onda.signals import AggregateSignal
from .core import Transform
from k_onda.central import DatasetSchema


class Aggregator:

    # TODO: do I want Aggregator to take an optional group by parameter
    # so that it can group a collection before aggregation?

    def __init__(self, method='mean'):
        self.method = method

    def __call__(self, input):

        from k_onda.sources import CollectionMap, Collection

        if isinstance(input, CollectionMap):
            return self._call_on_collection_map(input)
        
        elif isinstance(input, Collection):
            return self._call_on_collection(input)
        
        else:
            raise ValueError("Aggregator must be called on a Collection or a Grouped_Collection ")
         
        
    def _call_on_collection_map(self, collection_map):

        from ..sources import SignalMap

        transform = self._get_transform()

        return SignalMap(
            map={
                k: AggregateSignal(
                    inputs=collection.signals,
                    data_schema=collection.signals[0].data_schema,
                    transformer=self,
                    transform=transform
                )
                for k, collection in collection_map.items()
            }
        )
    
    def _call_on_collection(self, collection):
        transform = self._get_transform()
        # need to figure out why transform is passed signal, not data

        return AggregateSignal(
            inputs=collection.signals,
            transformer=self,
            transform=transform,
            data_schema=collection.signals[0].data_schema
        )

    def _get_transform(self):
        
        return Transform(self._apply)

    def _gather_datasets(self, signals):
        keys = signals[0].data.keys()
        data = {}
       
        for key in keys:
            arrays = []
            for signal in signals:
                arr = signal.data[key]
                arrays.append(arr)
    

            data[key] = xr.concat(
                arrays, dim=self.dim or 'members', combine_attrs='no_conflicts'
            )

        dataset = xr.Dataset(data)

        return dataset

    def _gather_arrays(self, signals):
        arrays = []
    
        for signal in signals:
            arr = signal.data
            arrays.append(arr)
           
        data = xr.concat(arrays, dim="members", combine_attrs="no_conflicts")

        return data
    
    def _gather(self, signals):
        if isinstance(signals[0].data, xr.Dataset):
            return self._gather_datasets(signals)
        return self._gather_arrays(signals)

    def _apply(self, signals):
        data = self._gather(signals)
        if self.method == 'concat':
            return data
        else:
            return getattr(data, self.method)(dim="members")
        
    

        
