import numpy as np
import pint
from collections.abc import Iterable
from collections import defaultdict
from copy import deepcopy

from k_onda.utils import is_unitful
from k_onda.central import types, SpanDimPair, DimBounds


DIM_DEFAULT_UNITS = {'time': 's', 'frequency': 'Hz'}


# TODO: condition filtering methods
# TODO: set algebra on Interval/MarkerSet
# TODO: Construction log 


@types.register
class Locus:

    name = 'locus'

    def __init__(self, dim,  units=None, ureg=None, conditions=None, metadim=None):
        self.dim = dim
        self.units = units
        self.ureg = ureg
        self.conditions = conditions or {}
        self.validate_conditions()
        self.metadim = metadim

    def validate_conditions(self):
        # TODO when I actually have docs the messages should be replaced with a nice link to the docs.
        for key in self.conditions.keys():
            if any([string in key for string in [
                'time', 'frequency', 'position', 'distance', 'sample', 'spike'
                ]]):
                raise ValueError("You can't include 'time', 'frequency', 'position', " \
                "'distance', 'sample', or 'spike' in a condition name.")
            if any([key == string for string in ['x', 'y']]):
                raise ValueError("'x' and 'y' are forbidden condition names.")
    
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


@types.register
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
            span=(self.value - window[0], self.value + window[1]),
            units=self.units, 
            anchor=self, 
            ureg=self.ureg, 
            index=self.index, 
            conditions=self.conditions
            )
    
    def to_intervals(self, window):
        return self.to_interval(window)


@types.register
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


@types.register
class Interval(Locus):

    marker_class = Marker
  
    def __init__(self, dim, span, parent_interval=None, anchor=None, units=None, 
                 ureg=None, index=None, conditions=None, metadim=None, label=None):
        super().__init__(dim, units=units, ureg=ureg, conditions=conditions)
        
        self.span = (span if isinstance(span, SpanDimPair) 
                     else SpanDimPair(pair=(self.w_units(span))))
        self.parent_interval = parent_interval
        self.anchor = anchor
        self.index = index
        self.label = label
        self.extent = self.span[1] - self.span[0]
        self.metadim = self.get_metadim(metadim)

        self.dim_bounds = DimBounds({self.dim: [self.span]})


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
    def generate_markers(self, spacing=None, offsets=None, count=None, positions=None, linspace=None, conditions=None):
        if len([arg for arg in (spacing, offsets, count, positions, linspace) if arg is not None]) != 1:
            raise ValueError("You must pass one and only one of `spacing`, `offsets`, `count`, " \
            "`positions`, `linspace` to generate_markers")
        # TODO: don't think this is right for the plural ones, offsets and positions
        lo, hi, extent, spacing, offsets, count, positions, linspace = [
            q.magnitude if q is not None else None 
            for q in [*self.span, self.extent, spacing, offsets, count, positions, linspace]
        ]
        if count is not None:
            places = np.arange(count) * (extent)/count + lo
        elif spacing is not None:
            places = np.arange(lo, hi, spacing)
        elif offsets is not None:
            places = [offset + lo for offset in offsets]
        elif positions is not None:
            places = [position for position in positions if lo <= position < hi]
        elif linspace is not None:
            places = np.linspace(lo, hi, linspace)

        return self.get_markers(places, conditions=conditions)
    
    @property
    def marker_set_class(self):
        return MarkerSet
    
    def get_markers(self, places, conditions=None):
        conditions = self.conditions | (conditions or {})
        return self.marker_set_class([
            self.marker_class(self.dim, place + self.span[0], index=i, units=self.units, conditions=self.conditions) 
            for i, place in enumerate(places)
            ], conditions=conditions)
    

@types.register         
class Epoch(Interval):

    name = 'epoch'
    marker_class = Event
    
    def __init__(self, session, onset, duration, units='s', parent_epoch=None, index=None, 
                 conditions=None, epoch_type=None, config=None, label=None):
        super().__init__(
            'time',
            (onset, onset + duration),
            index=index,
            parent_interval=parent_epoch,
            units=units, 
            ureg=session.experiment.ureg, 
            conditions=conditions,
            metadim='time',
            label = config.get('label') or label
            )
        self.session = session
        self.duration = self.w_units(duration)
        self.onset = self.w_units(onset)
        # epoch_type isn't the condition, but the key that allows you to identify
        # the epoch in the epoch config
        self.epoch_type = epoch_type  
        self.config = config
        self.parent_epoch = self.parent_interval
        self._events = defaultdict(list)

    @property
    def events(self):
        if not self._events:
            event_types = self.config.get('events', [])
            for event_type in event_types:
                config = self.session.experiment.events_config.get(event_type)
                if config is None: 
                    raise ValueError(
                        f"Events not configured for epoch of type {self.epoch_type} \
                        and events of type {event_type}")
                for key, val in config.items():
                    config[key] = self.w_units(val).to('s')
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
    
    def get_markers(self, markers, conditions=None): 
        conditions = self.conditions | (conditions or {})
        return ([Event(
            self.session, place, index=i, epoch=self, conditions=self.conditions) 
            for i, place in enumerate(markers)
            ])
     

@types.register
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


@types.register
class LocusSet(Locus):

    # TODO put set algebra here

    def __init__(self, loci, conditions=None, metadim=None):
        self.conditions = conditions or {}
        self._loci = loci
        self.validate(loci)
        self.metadim = metadim
    
    def __iter__(self):
        return iter(self.loci)
    
    def __len__(self):
        return len(self.loci)
    
    @property
    def loci(self):
        return self._loci
    
    def append(self, locus):
        self.validate([locus])
        self._loci.append(locus)

    def extend(self, loci):
        self.validate(loci)
        self._loci.extend(loci)

    def validate(self, loci):
        if self.loci and not all([type(locus) == type(self.loci[0]) for locus in loci]):
            raise ValueError(f"All loci in a {self.__class__.__name__} must be of the same type.")
    
    def conditions_are_met(self, conditions):
        for key, value in conditions.items():
            if key in self.conditions and value != self.conditions[key]:
                return False
        return True
                
    def where(self, conditions=None, **kwargs):
        if conditions is None:
            conditions = {}
        conditions.update(kwargs)
        loci = [locus for locus in self.loci if self.conditions_are_met(conditions)]
        if not len(loci):
            raise ValueError(f"Selection of {conditions} resulted in a length 0 {self.__class__.__name__}")
        return loci


@types.register
class MarkerSet(LocusSet):


    def __init__(self, places=None, dim=None, markers=None, conditions=None, units=None, ureg=None, metadim=None):
        if places is None and markers is None:
            raise ValueError("One of places or markers must not be None")
        if markers is None and dim is None:
            raise ValueError("You must supply `dim` if you're not passing markers that already have one.")
        self.places = None
        if markers is None:
            self.markers = self.markers_from_places()
        self._loci = markers
        self.dim = dim or self.markers[0].dim
        self.conditions = conditions or {}
        self.units = units
        self.ureg = ureg
        self.metadim = None

    @property
    def markers(self):
        return self._loci
    
    def markers_from_places(self):
        self.markers = [
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
        return IntervalSet(self.dim, intervals=intervals, ureg=self.ureg, units=self.units, conditions=self.conditions, metadim=self.metadim)


@types.register
class EventSet(MarkerSet):
    def __init__(self, session, events=None, places=None, conditions=None, units='s'):
        super().__init__(
            markers=events,
            places=places, 
            dim='time',
            conditions=conditions, 
            units=units, 
            ureg=session.experiment.ureg)
        self.session = session
        self.metadim = 'time'

    @property
    def events(self):
        return self.markers
    
    @property
    def to_epoch_set(self, window):
        return super().to_intervals(window)


@types.register
class IntervalSet(LocusSet):

    locus_class = Interval

    def __init__(self, dim, spans=None, intervals=None, ureg=None, units=None, conditions=None, metadim=None):
        if spans is None and intervals is None:
            raise ValueError("One of intervals or spans must not be None")
        super().__init__(loci=intervals, conditions=conditions, metadim=metadim)
        self.dim = dim
        self.spans = spans
        self.ureg = ureg
        self.units = units
        self.metadim = metadim
        self.conditions = conditions or {}

        if intervals is None:
            self.intervals_from_spans()

        self.dim_bounds = DimBounds(
            {self.dim: [interval.dim_bounds[dim][0] for interval in self.intervals]}
            )
      

    @property
    def intervals(self):
        return self._loci

    def intervals_from_spans(self):
        self._loci = [
            self.locus_class(
                self.dim, 
                span, 
                units=self.units, 
                index=i, 
                ureg=self.ureg, 
                conditions=self.conditions,
                metadim=self.metadim) 
            for i, span in enumerate(self.spans)]


@types.register
class EpochSet(IntervalSet):

    locus_class = Epoch

    def __init__(self, 
                 session, 
                 spans=None, 
                 epochs=None, 
                 conditions=None, 
                 units='s'):
        self.session = session
        super().__init__(
            'time', 
            spans=spans,
            intervals=epochs, 
            conditions=conditions, 
            units=units, 
            ureg=session.experiment.ureg,
            metadim='time')
        
    @property
    def epochs(self):
        return self._loci
        
    def epochs_from_spans(self):   
        self._loci = [
            Epoch(
                self.session, 
                span[0],
                span[1] - span[0], 
                units=self.units, 
                index=i, 
                conditions=self.conditions) 
            for i, span in enumerate(self.spans)]

    def intervals_from_spans(self):
        return self.epochs_from_spans()
    
    def validate_type(self, epochs):
        super().validate(epochs)
        if not all([isinstance(epoch, Epoch) for epoch in epochs]):
            raise ValueError("All members of an EpochSet must be of type Epoch.")

    
        
        





      

