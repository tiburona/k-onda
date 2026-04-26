from copy import deepcopy
import pint

from k_onda.mixins import DictDelegator


DIM_DEFAULT_UNITS = {'time': 's', 'frequency': 'Hz'}


class DimPair:
    def __init__(self, pair=None, units=None):
        if units is None and any([bound is None for bound in pair]):
            raise ValueError("You must define either bounds or units")
        self.units = units
        if pair is None:
            self.pair = [0 * units, 0 * units]
        else:
            self.pair = pair
       
    def __iter__(self):
        return iter(self.pair)
    
    def __getitem__(self, idx):
        return (self.pair)[idx]
    
    def __add__(self, other):
        self.validate_add(other)
        return DimPair(pair=(self[0] + other[0], self[1] + other[1]))
    
    def __iadd__(self, other):
        self.validate_add(other)
        return DimPair(pair=(self[0] + other[0], self[1] + other[1]))
    
    def validate_add(self, other):
        if all([isinstance(dp, SpanDimPair) for dp in (self, other)]):
            raise(ValueError("Don't add two SpanDimPairs; this operation is for " \
            "widening bounds."))
    

class SpanDimPair(DimPair):
    pass


   
class PadDimPair(DimPair):
    pass


class DimBounds(DictDelegator):

    _delegate_attr = '_dim_bounds' 

    def __init__(self, dim_pair_map=None, dim_pair_type='pad', units=None, 
                 metadim_of=None):
        self._dim_bounds = dict(dim_pair_map) if dim_pair_map else {}
        self._dim_pair_type = PadDimPair if dim_pair_type == 'pad' else SpanDimPair
        self.units = units
        self._metadim_of = metadim_of
  
    def __missing__(self, dim):
       
        if dim in DIM_DEFAULT_UNITS:
            units = DIM_DEFAULT_UNITS[dim]
            default = self._dim_pair_type(units=pint.application_registry(units))
            self._dim_bounds[dim] = default
            return default
        else:
            raise KeyError(f"dim {dim} not in DimBounds")
    
    def __and__(self, other):
        # todo, define this
        pass

    def __add__(self, other):
        self_copy = deepcopy(self)
        return self_copy.merge(other)

    def __iadd__(self, other):
        return self.merge(other)
    
    def _equivalent(self, key, other_key):
        if key == other_key: return True
        if key is None and other_key is None: return False
        return (self._metadim_of and
                self._metadim_of(key) == self._metadim_of(other_key))
    
    def are_equivalent(self, my_dim, their_dim, strict):
        return (
            (not strict and self._equivalent(my_dim, their_dim)) or
            (their_dim == my_dim)
            )
       
    def _per_dim_add(self, other, my_dim, their_dim):
        if isinstance(self[my_dim], DimPair):
            if isinstance(other[their_dim], DimPair):
                self[my_dim] += other[their_dim]
            else:
                other_dim = deepcopy(other[their_dim])
                for bounds in other_dim:
                    bounds += self[my_dim]
                self[my_dim] = other_dim
            
        else:
            if isinstance(other[their_dim], DimPair):
                for bounds in self[my_dim]:
                    bounds += other[their_dim]
            else:
                if len(self[my_dim]) == len(other[their_dim]):
                    for i, bounds in enumerate(self[my_dim]):
                        bounds += other[their_dim][i]
                else:
                    raise ValueError(
                        f"{self} and {other} have incompatible dimensions"
                        )

    def plus(self, other, inclusive=True, strict=False):
        for their_dim in other:
            did_match=False
            for my_dim in self:
                if self.are_equivalent(my_dim, their_dim, strict):
                    did_match = True
                    self._per_dim_add(other, my_dim, their_dim)
            if not did_match:
                if inclusive:
                    self[their_dim] = other[their_dim]
        return self
    
    def merge(self, other):
        return self.plus(other, inclusive=True)
    
    def accumulate(self, other):
        return self.plus(other, inclusive=False)
    
    def to_array_of_dicts(self):
       
        if isinstance(next(iter(self._dim_bounds.values())), DimPair):
            return [self._dim_bounds]
        else:
            # I have a dictionary of lists
            # and I want a list of dictionaries
            n = len(list(self._dim_bounds.values())[0])
            return [
                {dim: bounds[i] for dim, bounds in self._dim_bounds.items()} 
                for i in range(n)
                ]