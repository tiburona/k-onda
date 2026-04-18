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

    def __init__(self, dim_pair_map=None, dim_pair_type= 'pad', units=None):
        self._dim_bounds = dict(dim_pair_map) if dim_pair_map else {}
        self._dim_pair_type = PadDimPair if dim_pair_type == 'pad' else SpanDimPair
        self.units = units
  
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
        
    def _plus(self, other, inclusive=True):
        for dim in other:
            if dim in self:
                if isinstance(self[dim], DimPair):
                    if isinstance(other[dim], DimPair):
                        self[dim] += other[dim]
                    else:
                        other_dim = deepcopy(other[dim])
                        for bounds in other_dim:
                            bounds += self[dim]
                        self[dim] = other_dim
                        
                else:
                    if isinstance(other[dim], DimPair):
                        for bounds in self[dim]:
                            bounds += other[dim]
                    else:
                        if len(self[dim]) == len(other[dim]):
                            for i, bounds in enumerate(self[dim]):
                                bounds += other[dim][i]
                        else:
                            raise ValueError(
                                f"{self} and {other} have incompatible dimensions"
                                )
            else:
                if inclusive:
                    self.__setitem__(dim, other[dim])

        return self
    
    def merge(self, other):
        return self._plus(other, inclusive=True)
    
    def accumulate(self, other):
        return self._plus(other, inclusive=False)
    
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