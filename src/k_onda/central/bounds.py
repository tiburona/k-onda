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

    def __repr__(self):
        return f"{self[0]}, {self[1]}"
       
    def __iter__(self):
        return iter(self.pair)
    
    def __getitem__(self, idx):
        return (self.pair)[idx]

    @property
    def lo(self):
        return self[0]

    @lo.setter
    def lo(self, value):
        self.pair = (value, self.hi)

    @property
    def hi(self):
        return self[1]

    @hi.setter
    def hi(self, value):
        self.pair = (self.lo, value)
    
    def __add__(self, other):
        self.validate_add(other)
        return DimPair(pair=(self[0] + other[0], self[1] + other[1]))
    
    def __iadd__(self, other):
        self.validate_add(other)
        self.pair = (self[0] + other[0], self[1] + other[1])
        return self
    
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

    def __repr__(self):
        return repr(self._dim_bounds)
  
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

    def _value_for_bounds(self, bounds, side):
        if isinstance(bounds, DimPair):
            return getattr(bounds, side)
        return [getattr(bound, side) for bound in bounds]

    def _side_values(self, side):
        values_by_dim = {
            dim: self._value_for_bounds(bounds, side)
            for dim, bounds in self._dim_bounds.items()
            }
        if len(values_by_dim) == 1:
            return next(iter(values_by_dim.values()))
        return values_by_dim

    @property
    def lo(self):
        return self._side_values('lo')

    @property
    def hi(self):
        return self._side_values('hi')

    def _add_pair_side(self, target, source, side):
        target.validate_add(source)
        setattr(target, side, getattr(target, side) + getattr(source, side))

    def _add_pair(self, target, source):
        target += source

    def _per_dim_apply(
            self,
            other,
            my_dim,
            their_dim,
            pair_op,
            expand_list_source_from='other',
            ):
        my_bounds = self[my_dim]
        their_bounds = other[their_dim]

        if isinstance(my_bounds, DimPair):
            if isinstance(their_bounds, DimPair):
                pair_op(my_bounds, their_bounds)
            elif expand_list_source_from == 'self':
                self[my_dim] = []
                for their_bound in their_bounds:
                    bounds = deepcopy(my_bounds)
                    pair_op(bounds, their_bound)
                    self[my_dim].append(bounds)
            else:
                their_bounds = deepcopy(their_bounds)
                for bounds in their_bounds:
                    pair_op(bounds, my_bounds)
                self[my_dim] = their_bounds
        else:
            if isinstance(their_bounds, DimPair):
                for bounds in my_bounds:
                    pair_op(bounds, their_bounds)
            elif len(my_bounds) == len(their_bounds):
                for i, bounds in enumerate(my_bounds):
                    pair_op(bounds, their_bounds[i])
            else:
                raise ValueError(
                    f"{self} and {other} have incompatible dimensions"
                    )

    def _per_dim_add(self, other, my_dim, their_dim):
        self._per_dim_apply(other, my_dim, their_dim, self._add_pair)

    def _per_dim_add_side(self, other, my_dim, their_dim, side):
        def add_pair_side(target, source):
            self._add_pair_side(target, source, side)

        self._per_dim_apply(
            other,
            my_dim,
            their_dim,
            add_pair_side,
            expand_list_source_from='self',
            )

    def _apply_to_matching_dims(
            self,
            other,
            per_dim_op,
            inclusive=False,
            strict=False,
            require_match=False,
            ):
        for their_dim in other:
            did_match = False
            for my_dim in list(self):
                if self.are_equivalent(my_dim, their_dim, strict):
                    did_match = True
                    per_dim_op(other, my_dim, their_dim)
            if not did_match:
                if inclusive:
                    self[their_dim] = other[their_dim]
                elif require_match:
                    raise ValueError(f"dim {their_dim} not found in {self}")
        return self

    def add_lo(self, other, strict=False):
        return self._apply_to_matching_dims(
            other,
            lambda other, my_dim, their_dim: self._per_dim_add_side(
                other, my_dim, their_dim, 'lo'),
            strict=strict,
            require_match=True,
            )

    def add(self, other, inclusive=True, strict=False):
        return self._apply_to_matching_dims(
            other,
            self._per_dim_add,
            inclusive=inclusive,
            strict=strict,
            )
    
    def merge(self, other):
        return self.add(other, inclusive=True)
    
    def accumulate(self, other):
        return self.add(other, inclusive=False)
    
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


class DimOffset:
    _delegate_attr = '_dim_offset' 

    def __init__(self, dim_offset_map=None,  units=None, 
                 metadim_of=None):
        self._dim_offset = dict(dim_offset_map) if dim_offset_map else {}
        self.units = units
        self._metadim_of = metadim_of