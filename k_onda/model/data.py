import xarray as xr

from k_onda.core import Base
from .aggregates import Aggregates
from k_onda.utils import operations


class TransformRegistryMixin:
    TRANSFORMS: dict[str, tuple] = {}  # default empty


class Data(Base, Aggregates, TransformRegistryMixin):    

    TRANSFORMS: dict[str, tuple] = {}

    def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.parent = None

    @property
    def name(self):
        return self._name
    
    def fetch_opts(self, list_of_opts=None):
        if list_of_opts is not None:
            return (self.calc_opts.get(opt) for opt in list_of_opts)
        
    def include(self, check_ancestors=True):
        return self.select(self.criteria, check_ancestors=check_ancestors)
    
    def active(self):
        return self.include() and self in self.parent.children
    
    @property
    def unique_id(self):
        if not hasattr(self, '_unique_id'):
            return self.identifier
        else:
            return self._unique_id

    
    @property
    def included_children(self):
        
        if hasattr(self, 'children'):
            return [child for child in self.children if child.include()]
        else:
            return None
    
    @property
    def has_children(self):
        return bool(len(self.children))
    
    @property
    def has_grandchildren(self):
        return any([child.has_children for child in self.children])
    
    @property
    def parent_identifier(self):
        try:
            return self.parent.identifier
        except AttributeError:
            return None
        
    @property
    def grandparent_identifier(self):
        try:
            return self.ancestors[-3].identifier
        except IndexError:
            return None
        
    def get_child_by_identifier(self, id):
        return [child for child in self.children if child.identifier == id][0]
    
    @property
    def experiment_wise_index(self):
        if self.name in ['period', 'event']:
            name = f'{self.calc_type}_{self.name}'
        else:
            name = self.name
        return getattr(self.experiment, f'all_{name}s').index(self)
        
    def sort(self, sort, items):
        if not sort:
            return items
        sort_key, order = sort
        sorted_lst = sorted(
            items, 
            key=lambda x: getattr(x, sort_key), 
            reverse=(order == 'descending'))
        return sorted_lst

    def sort_children(self, children):
        sort = self.calc_opts.get('sort', {}).get(self.name)
        if not sort:
            return children
        else:
            sort_key, order = sort
            sorted_children = sorted(
                children, 
                key=lambda x: getattr(x, sort_key), 
                reverse=(order == 'descending'))
            return sorted_children
            
    def select(self, criteria, check_ancestors=False):
           
        if not check_ancestors and self.name not in criteria:
            return True
              
        for obj in (self.ancestors if check_ancestors else [self]):
            if obj.name not in criteria:
                continue
            obj_filters = criteria[obj.name]
            for attr in obj_filters:
                if hasattr(obj, attr):
                    object_value = getattr(obj, attr)
                    operation_symbol, target_value = obj_filters[attr]
                    function = operations[operation_symbol]
                    if not function(object_value, target_value):
                        return False
        return True
    
    @property
    def ancestors(self):
        if self.name == 'experiment':
            return [self]
        if hasattr(self, 'parent'):
            return self.parent.ancestors + [self]
        
    @property
    def descendants(self):
        return self.get_descendants
    
    @property
    def index(self):
        return self.hierarchy.index(self.name)
        
    @property
    def hierarchy(self):
        if self.kind_of_data == 'spike':
            hierarchy = ['experiment', 'animal', 'unit', 'period', 'event']
        elif self.kind_of_data == 'lfp':
            if self.calc_type in ['amp_xcorr', 'lag_of_max_corr']:
                hierarchy = ['experiment', 'animal', 'amp_xcorr_calculator', 'amp_xcorr_event']
                if self.calc_opts.get('validate_events'):
                    hierarchy.insert(3, 'amp_xcorr_segment')
            elif self.calc_type == 'coherence':
                hierarchy = ['experiment', 'animal', 'coherence_calculator', 'coherence_event']
                if self.calc_opts.get('validate_events'):
                    hierarchy.insert(3, 'coherence_segment')
            else:
                hierarchy = ['experiment', 'animal', 'period', 'event']
        elif self.kind_of_data == 'mrl':
            hierarchy = ['experiment', 'animal', 'unit', 'mrl_calculator']
        else:
            raise ValueError('Unknown kind of data')
        if self.experiment.all_groups:
            hierarchy.insert(1, 'group')
        return hierarchy
 
    @property
    def sampling_rate(self):
        if self.name == 'experiment':
            return self._sampling_rate
        else:
            return self.experiment.sampling_rate
        
    @property
    def lfp_sampling_rate(self):
        if self.name == 'experiment':
            return self._lfp_sampling_rate
        else:
            return self.experiment.lfp_sampling_rate
        
    def get_descendants(self, stop_at=None, descendants=None, all=False):
   
        if descendants is None:
            descendants = []

        if self.name == stop_at or not hasattr(self, 'children'):
            descendants.append(self)
        else:
            if all:
                descendants.append(self)

            for child in self.children:
                child.get_descendants(descendants=descendants, stop_at=stop_at, all=all)

        return descendants
    
    def find_attr_in_ancestors(self, attr):
        for obj in reversed(self.ancestors):
            if hasattr(obj, attr):
                return getattr(obj, attr)
        return None

    @property
    def has_reference(self):
        return hasattr(self, 'reference') and self.reference is not None
    
    @classmethod
    def all_transforms(cls) -> dict:
        """
        Merge TRANSFORMS dicts from cls and all base classes.
        Later classes in the MRO override earlier ones on key conflicts.
        """
        merged: dict = {}
        # Walk MRO from base → subclass so subclasses win
        for base in reversed(cls.mro()):
            d = getattr(base, "TRANSFORMS", None)
            if d:
                merged.update(d)
        return merged

    def transforms(self) -> dict:
        # instance-facing helper
        return type(self).all_transforms()
    
    def get_transform_pair_for_da(self, da):
        # Only DataArrays that explicitly opt in via a key get transformed
        if not isinstance(da, xr.DataArray):
            return (None, None)
        key = da.attrs.get("transform_key", None)
        if key is None:
            return (None, None)
        registry = self.transforms() if hasattr(self, "transforms") else {}
        return registry.get(key, (None, None))

    def to_linear_space(self, data):
        if isinstance(data, xr.Dataset):
            ds = data.copy()
            for name in ds.data_vars:
                ds[name] = self.to_linear_space(ds[name])
            return ds

        transform, _ = self.get_transform_pair_for_da(data)
        if transform is None:
            return data

        space = getattr(data, "attrs", {}).get("space", "raw")
        if space == "z":
            return data
        if space in ("raw", "final"):
            out = transform(data)
            out.attrs.update(data.attrs)
            out.attrs["space"] = "z"
            return out
        return data

    def to_final_space(self, data):
        if isinstance(data, xr.Dataset):
            ds = data.copy()
            for name in ds.data_vars:
                ds[name] = self.to_final_space(ds[name])
            return ds

        _, back_transform = self.get_transform_pair_for_da(data)
        if back_transform is None:
            return data

        space = getattr(data, "attrs", {}).get("space", "raw")
        if space in ("final", "raw"):
            return data
        if space == "z":
            out = back_transform(data)
            out.attrs.update(data.attrs)
            out.attrs["space"] = "final"
            return out
        return data
    
