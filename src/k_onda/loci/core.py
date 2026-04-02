import numpy as np
import pint
from collections.abc import Iterable
from collections import defaultdict
from copy import deepcopy

from k_onda.utils import is_unitful
from k_onda.mixins import DictDelegator


DIM_DEFAULT_UNITS = {'time': 's', 'frequency': 'Hz'}


# TODO: condition filtering methods
# TODO: set algebra on Interval/MarkerSet
# TODO: Construction log 


class DimPair:
    def __init__(self, units=None, lo=None, hi=None):
        if units is None and any([bound is None for bound in [lo, hi]]):
            raise ValueError("You must define either bounds or units")
        self.units = units
        self.lo = lo if lo is not None else 0 * units
        self.hi = hi if hi is not None else 0 * units


    def __iter__(self):
        return iter((self.lo, self.hi))
    
    def __add__(self, other):
        self.validate_add(other)
        return DimPair(self.lo + other.lo, self.hi + other.hi)
    
    def __iadd__(self, other):
        self.validate_add(other)
        self.lo = self.lo + other.lo
        self.hi = self.hi + other.hi
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

    def __init__(self, dim_pair_map=None, dim_pair_type= 'pad'):
        self._dim_bounds = dict(dim_pair_map) if dim_pair_map else {}
        self._dim_pair_type = PadDimPair if dim_pair_type == 'pad' else SpanDimPair
  
    def __missing__(self, dim):
       
        if dim in DIM_DEFAULT_UNITS:
            units = DIM_DEFAULT_UNITS[dim]
            default = self._dim_pair_type(units)
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
        
            
        


class Locus:

    name = 'locus'

    def __init__(self, dim,  units=None, ureg=None, conditions=None, metadim=None):
        self.dim = dim
        self.units = units
        self.ureg = ureg
        self.conditions = self.validate_conditions(conditions)
        self.metadim = metadim

    def validate_conditions(self, conditions):
        # TODO when I actually have docs the messages should be replaced with a nice link to the docs.
        for key in conditions.keys():
            if any([string in key for string in [
                'time', 'frequency', 'position', 'distance', 'sample', 'spike'
                ]]):
                raise ValueError("You can't include 'time', 'frequency', 'position', " \
                "'distance', 'sample', or 'spike' in a condition name.")
            if any([key == string for string in ['x', 'y']]):
                raise ValueError("'x' and 'y' are forbidden condition names.")
            
        return conditions
    
    def w_units(self, value):
        if is_unitful(value):
            return value
        if self.units is None:
            if self.dim in DIM_DEFAULT_UNITS:
                self.units = DIM_DEFAULT_UNITS[self.dim]
            else:
                raise ValueError("units were not provided to Locus")
        if self.ureg is None:
            raise ValueError("ureg was not provided to locus.")
        if isinstance(value, Iterable):
            return [v * self.ureg(self.units) for v in value]
        else:
            return value * self.ureg(self.units)



class Marker(Locus):

    name = 'marker'

    def __init__(self, dim, value, index=None, parent_interval=None, units=None, ureg=None, conditions=None):
        super().__init__(dim, units=units, ureg=ureg, conditions=conditions)
        self.value = self.w_units(value)
        self.index = index
        self.parent_interval = parent_interval

    def to_interval(self, window):
        return Interval(
            self.dim, 
            (self.value - window[0], self.value + window[1]),
            units=self.units, 
            anchor=self, 
            ureg=self.ureg, 
            index=self.index, 
            conditions=self.conditions
            )
    
    def to_intervals(self, window):
        return self.to_interval(window)


class Event(Marker):

    name = 'event'

    def __init__(self, session, value, index=None, epoch=None, units='s', conditions=None):
        super().__init__(
            'time', 
            value, 
            index, 
            epoch,
            units, 
            ureg=session.experiment.ureg, 
            conditions=conditions)
        self.session = session
        self.parent_epoch = epoch
        self.metadim = 'time'


class Interval(Locus):

    marker_class = Marker
  
    def __init__(self, dim, span, parent_interval=None, anchor=None, units=None, 
                 ureg=None, index=None, conditions=None, metadim=None):
        super().__init__(dim, units=units, ureg=ureg, conditions=conditions)
        self.span = SpanDimPair(lo=self.w_units(span)[0], hi=self.w_units(span)[1]) 
        self.parent_interval = parent_interval
        self.anchor = anchor
        self.index = index
        self.extent = self.span.hi - self.span.lo
        self.metadim = self.get_metadim(metadim)

        self.dim_bounds = DimBounds({self.dim: span})
        self.metadim_bounds = DimBounds({self.metadim: span})

        self.metadim_map = {self.metadim: self.dim}
        self.dim_map = {self.dim: self.metadim}

    def get_metadim(self, metadim):
        if metadim is not None:
            return metadim
        if self.dim in DIM_DEFAULT_UNITS:
            return self.dim
        else:
            dim_parts = self.dim.split('_')
            if dim_parts[-1] in DIM_DEFAULT_UNITS or len(dim_parts) < 3:
                return dim_parts[-1]
            else:
                raise ValueError("Metadim was not provided and cannot be inferred.")

    # TODO: it should probably also be an option to pass 'absolute'
    def generate_markers(self, spacing=None, offsets=None, count=None, positions=None, linspace=None):
        if len([arg for arg in (spacing, offsets, count, positions, linspace) if arg is not None]) != 1:
            raise ValueError("You must pass exactly 1 keyword argument to generate_markers")
        if count is not None:
            places = np.arange(count) * (self.extent)/count + self.span[0]
        elif spacing is not None:
            places = np.arange(self.span[0], self.span[1], spacing)
        elif offsets is not None:
            places = [offset + self.span[0] for offset in offsets]
        elif positions is not None:
            places = [position for position in positions if self.span[0] <= position < self.span[1]]
        elif linspace is not None:
            places = np.linspace(self.span[0], self.span[1], linspace)

        return self.get_markers(places)
    
    @property
    def marker_set_class(self):
        return MarkerSet
    
    def get_markers(self, places):
        return self.marker_set_class([
            self.marker_class(self.dim, place + self.span[0], index=i, units=self.units, conditions=self.conditions) 
            for i, place in enumerate(places)
            ], conditions=self.conditions)
    
    def metadim_to_dim(self, metadim):
        return self.metadim_map[metadim]
    
            
class Epoch(Interval):

    name = 'epoch'
    marker_class = Event
    
    def __init__(self, session, onset, duration, units='s', parent_epoch=None, index=None, 
                 conditions=None, epoch_type=None):
        super().__init__(
            'time',
            (onset, onset + duration),
            index=index,
            parent_interval=parent_epoch,
            units=units, 
            ureg=session.experiment.ureg, 
            conditions=conditions,
            metadim='time'
            )
        self.session = session
        self.duration = self.w_units(duration)
        self.onset = self.w_units(onset)
        # epoch_type isn't the condition, but the key that allows you to identify
        # the epoch in the epoch config
        self.epoch_type = epoch_type  
        self.parent_epoch = self.parent_interval
        self._events = defaultdict(list)

    @property
    def events(self):
        if not self._events:
            event_types = (self.session.config
                      .get('epochs', {})
                      .get(self.epoch_type, {})
                      .get('events'))
            for event_type in event_types:
                config = self.session.experiment.events_config.get(event_type)
                if config is None: 
                    raise ValueError(
                        f"Events not configured for epoch of type {self.epoch_type} \
                        and events of type {event_type}")
                self._events[event_type] = self.generate_events(**config)
        
        return [event for event_list in self._events.values() for event in event_list]
       
    def generate_events(
            self, 
            spacing=None, 
            offsets=None, 
            count=None, 
            positions=None, 
            linspace=None, 
            conditions=None
            ):
        return self.generate_markers(spacing, offsets, count, positions, linspace, conditions)
    
    @property
    def marker_set_class(self):
        return EventSet
    
    def get_markers(self, markers): 

        return ([Event(
            self.session, place, index=i, parent_epoch=self, conditions=self.conditions) 
            for i, place in enumerate(markers)
            ])
     

class FrequencyBand(Interval):

    name = 'frequency_band'

    def __init__(self, span, index=None, ureg=None, conditions=None):
        super().__init__(
            'frequency', 
            span, 
            units='Hz', 
            ureg=ureg, 
            index=index, 
            conditions=conditions,
            metadim='frequency')


class LocusSet(Locus):

    # TODO put set algebra here

    def __init__(self, loci, conditions=None, metadim=None):
        self.conditions = conditions or {}
        self.validate([loci])
        self.loci = loci
        self.metadim = metadim
    
    @property
    def dim_bounds(self):
        dim = self.loci[0].dim
        bounds = [locus.dim_bounds[dim] for locus in self.loci]
        return {dim: bounds}
    
    def __iter__(self):
        return iter(self.loci)
    
    def __len__(self):
        return len(self.loci)
    
    def append(self, locus):
        self.validate([locus])
        self.loci.append(locus)

    def extend(self, loci):
        self.validate(loci)
        self.loci.extend(loci)

    def validate(self, loci):
        if len(self.loci) and not all([type(locus) == type(self.loci[0]) for locus in loci]):
            raise ValueError(f"All loci in a {self.__class__.__name__} must be of the same type.")

        # maybe this is not helpful.
        # if self.conditions and not all([locus.conditions == self.conditions for locus in loci]):
        #     raise ValueError(f"If conditions is defined on a {self.__class__.__name__} " \
        #                      "all members must have the same conditions.")
    
    def conditions_are_met(self, conditions):
        for key, value in conditions.items():
            if key in self.conditions and value != self.conditions[key]:
                return False
        return True
                
    def where(self, conditions):
        loci = [locus for locus in self.loci if self.conditions_are_met(conditions)]
        if not len(loci):
            raise ValueError(f"Selection of {conditions} resulted in a length 0 {self.__class__.__name__}")


class MarkerSet(LocusSet):


    def __init__(self, places=None, dim=None, markers=None, conditions=None, units=None, ureg=None, metadim=None):
        if places is None and markers is None:
            raise ValueError("One of places or markers must not be None")
        if markers is None and dim is None:
            raise ValueError("You must supply `dim` if you're not passing markers that already have one.")
        self.markers = markers
        self.places = None
        self.dim = dim or self.markers[0].dim
        self.conditions = conditions
        self.units = units
        self.ureg = ureg
        self.metadim = None

    @property
    def loci(self):
        return self.markers
    
    def markers_from_places(self):
        self.intervals = [
            self.marker_class(
                self.dim, 
                place, 
                units=self.units, 
                index=i, 
                ureg=self.ureg, 
                conditions=self.conditions,
                metadim=self.metadim) 
            for i, place in enumerate(self.places)]
        
    def to_intervals(self, window):
        return self.to_interval_set(window)
    
    def to_interval_set(self, window):
        intervals = [marker.to_interval(window) for marker in self.markers]
        return IntervalSet(self.dim, intervals, self.conditions, self.metadim)


class EventSet(MarkerSet):
    def __init__(self, session, events, conditions=None, units='s'):
        super().__init__(
            events, 
            conditions=conditions, 
            units=units, 
            ureg=session.experiment.ureg)
        self.dim = 'time'
        self.session = session
        self.metadim = 'time'

    @property
    def events(self):
        return self.markers
    
    @property
    def to_epoch_set(self, window):
        return super().to_intervals(window)


class IntervalSet(LocusSet):

    locus_class = Interval

    def __init__(self, dim, spans=None, intervals=None, ureg=None, units=None, conditions=None, metadim=None):
        if spans is None and intervals is None:
            raise ValueError("One of intervals or spans must not be None")
        self.dim = dim
        self.spans = spans
        self.intervals = intervals
        self.conditions = conditions
        self.ureg = ureg
        self.units = units
        self.metadim = metadim

        if self.intervals is None:
            self.intervals_from_spans()

        self.dim_bounds = DimBounds(
            {self.dim: [interval.dim_bounds[dim] for interval in self.intervals]}
            )
        self.metadim_bounds = DimBounds(
            {self.metadim: [interval.dim_bounds[dim] for interval in self.intervals]}
        )

        self.metadim_map = {self.metadim: self.dim}
        self.dim_map = {self.dim: self.metadim}

        

    @property
    def loci(self):
        return self.intervals

    def intervals_from_spans(self):
        self.intervals = [
            self.locus_class(
                self.dim, 
                span, 
                units=self.units, 
                index=i, 
                ureg=self.ureg, 
                conditions=self.conditions,
                metadim=self.metadim) 
            for i, span in enumerate(self.spans)]
        
    def metadim_to_dim(self, metadim):
        return self.metadim_map[metadim]


class EpochSet(IntervalSet):

    locus_class = Epoch

    def __init__(self, 
                 session, 
                 spans=None, 
                 epochs=None, 
                 conditions=None, 
                 units='s'):
        super().__init__(
            'time', 
            spans=spans,
            intervals=epochs, 
            conditions=conditions, 
            units=units, 
            ureg=session.experiment.ureg,
            metadim='time')
        self.session = session

    @property
    def epochs(self):
        return self.loci
        
    def epochs_from_spans(self):   
        self.loci = [
            Epoch(
                self.session, 
                span[0],
                span[1] - span[0], 
                units=self.units, 
                index=i, 
                conditions=self.conditions) 
            for i, span in enumerate(self.spans)]

    def intervals_from_spans(self):
        return self.epochs_from_intervals()
    
    def validate_type(self, epochs):
        super().validate(epochs)
        if not all([isinstance(epoch, Epoch) for epoch in epochs]):
            raise ValueError("All members of an EpochSet must be of type Epoch.")

    
        
        





      

