import numpy as np
from collections.abc import Iterable
from collections import defaultdict

from k_onda.utils import is_unitful, wout_units
from k_onda.central import type_registry, SpanDimPair, DimBounds


DIM_DEFAULT_UNITS = {"time": "s", "frequency": "Hz"}


class LocusBase:

    def __init__(self, locus_type=None, metadim=None, conditions=None, inherited_conditions=None):
        self.locus_type = locus_type
        self.metadim = metadim
        self.conditions = conditions or {}
        self.inherited_conditions = inherited_conditions or {}
        self.validate_conditions()

    def validate_conditions(self):
        for key in self.conditions.keys():
            if any(
                [
                    string in key
                    for string in [
                        "time",
                        "frequency",
                        "position",
                        "distance",
                        "sample",
                        "spike",
                    ]
                ]
            ):
                raise ValueError(
                    "You can't include 'time', 'frequency', 'position', "
                    "'distance', 'sample', or 'spike' in a condition name."
                )
            if any([key == string for string in ["x", "y"]]):
                raise ValueError("'x' and 'y' are forbidden condition names.")

            if (key in self.inherited_conditions and 
                self.inherited_conditions[key] != self.conditions[key]):
                raise ValueError(f"Incompatible condition {key}: {self.conditions[key]}")
            
    @property
    def all_conditions(self):
        return self.inherited_conditions | self.conditions
    
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


@type_registry.register
class Locus(LocusBase):
    name = "locus"

    def __init__(self, dim, *, units=None, ureg=None, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.units = units
        self.ureg = ureg

    def conditions_are_met(self, conditions, strict=True):
        for key, value in conditions.items():
            if strict:
                if self.all_conditions.get(key) != value:
                    return False
            else:
                if key in self.conditions and value != self.conditions[key]:
                    return False
        return True



@type_registry.register
class Marker(Locus):
    name = "marker"

    def __init__(self, dim, value, *, index=None, parent_interval=None, **kwargs):
        super().__init__(dim, **kwargs)
        self.value = self.w_units(value)
        self.index = index
        self.parent_interval = parent_interval

    def to_interval(self, window):
        return Interval(
            self.dim,
            span=(self.value + window[0], self.value + window[1]),
            units=self.units,
            anchor=self,
            ureg=self.ureg,
            index=self.index,
            parent_interval=self.parent_interval,
            conditions=self.conditions,
            inherited_conditions=self.inherited_conditions,
            locus_type=self.locus_type
        )

    def to_intervals(self, window):
        return self.to_interval(window)


@type_registry.register
class Event(Marker):
    name = "event"

    def __init__(self, session, value, epoch=None, **kwargs):
        super().__init__(
            "time", 
            value, 
            parent_interval=epoch, 
            units="s", 
            ureg=session.experiment.ureg, 
            **kwargs
            )
        self.session = session
        self.parent_epoch = epoch
        self.metadim = "time"


@type_registry.register
class Interval(Locus):
    marker_class = Marker

    def __init__(
        self,
        dim,
        span,
        parent_interval=None,
        anchor=None,
        units=None,
        ureg=None,
        index=None,
        label=None,
        metadim=None,
        **kwargs
    ):
        super().__init__(dim, units=units, ureg=ureg, **kwargs)

        self.span = (
            span
            if isinstance(span, SpanDimPair)
            else SpanDimPair(pair=(self.w_units(span)))
        )
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
            dim_parts = self.dim.split("_")
            if dim_parts[-1] in DIM_DEFAULT_UNITS or len(dim_parts) < 3:
                return dim_parts[-1]
            else:
                raise ValueError("Metadim was not provided and cannot be inferred.")

    def get_markers(
        self,
        spacing=None,
        offsets=None,
        count=None,
        positions=None,
        linspace=None,
        conditions=None
     ):
        places = self.generate_marker_places(
            spacing=spacing, 
            offsets=offsets, 
            count=count, 
            positions=positions, 
            linspace=linspace
        )

        markers = self.build_markers(places, conditions=conditions)

        return markers

    def generate_marker_places(
        self,
        spacing=None,
        offsets=None,
        count=None,
        positions=None,
        linspace=None
    ):
        if (
            len(
                [
                    arg
                    for arg in (spacing, offsets, count, positions, linspace)
                    if arg is not None
                ]
            )
            != 1
        ):
            raise ValueError(
                "You must pass one and only one of `spacing`, `offsets`, `count`, "
                "`positions`, `linspace` to generate_markers"
            )
        lo, hi, extent, spacing, offsets, count, positions, linspace = [
            wout_units(q) if q is not None else None
            for q in [
                *self.span,
                self.extent,
                spacing,
                offsets,
                count,
                positions,
                linspace,
            ]
        ]
        if count is not None:
            places = np.arange(count) * (extent) / count + lo
        elif spacing is not None:
            places = np.arange(lo, hi, spacing)
            eps = np.finfo(float).eps * max(abs(lo), abs(hi), abs(spacing), 1)
            places = places[places < hi - eps] 
        elif offsets is not None:
            places = [offset + lo for offset in offsets]
        elif positions is not None:
            places = [position for position in positions if lo <= position < hi]
        elif linspace is not None:
            places = np.linspace(lo, hi, linspace)

        return places

    @property
    def marker_set_class(self):
        return MarkerSet

    def build_markers(self, places, conditions=None):
        return self.marker_set_class(
            markers = [
                self.marker_class(
                    self.dim,
                    place,
                    index=i,
                    units=self.units,
                    ureg=self.ureg,
                    conditions=conditions or {},
                    inherited_conditions = self.inherited_conditions | self.conditions
                )
                for i, place in enumerate(places)
            ],
            conditions=conditions,
        )


@type_registry.register
class Epoch(Interval):
    name = "epoch"
    marker_class = Event

    def __init__(
        self,
        session,
        onset,
        duration,
        units="s",
        parent_epoch=None,
        index=None,
        conditions=None,
        config=None,
        label=None,
        locus_type=None,
        key=None
    ):
        self.config = config or {}
        super().__init__(
            "time",
            (onset, onset + duration),
            index=index,
            parent_interval=parent_epoch,
            units=units,
            ureg=session.ureg,
            conditions=conditions,
            metadim="time",
            label=label or self.config.get("label"),
            locus_type=locus_type or self.config.get("epoch_type")
        )
        self.session = session
        self.duration = self.w_units(duration)
        self.onset = self.w_units(onset)
        self.parent_epoch = self.parent_interval
        self.epoch_key = key
        self._events = defaultdict(list)

    @property
    def events(self):
        if not self._events:
            event_keys = self.config.get("events", [])
            for event_key in event_keys:
                config = self.session.experiment.events_config.get(event_key)
                if config is None:
                    raise ValueError(
                        f"Events not configured for epoch of key {self.epoch_key} \
                        and events of type {event_key}"
                    )
                for key, val in config.items():
                    config[key] = self.w_units(val).to("s")
                self._events[event_key] = self.get_events(**config)

        return [event for event_list in self._events.values() for event in event_list]

    def get_events(
        self,
        spacing=None,
        offsets=None,
        count=None,
        positions=None,
        linspace=None, 
        conditions=None
        ):

        places = self.generate_marker_places(spacing, offsets, count, positions, linspace)
        events = self.build_events(
            places, 
            conditions=conditions, 
            inherited_conditions=self.inherited_conditions | self.conditions
            )
        return events

    @property
    def marker_set_class(self):
        return EventSet

    def build_events(self, places, **kwargs):
        
        return [
            Event(self.session, place, index=i, epoch=self, **kwargs)
            for i, place in enumerate(places)
        ]


@type_registry.register
class FrequencyBand(Interval):
    name = "frequency_band"

    def __init__(self, span, index=None, ureg=None, conditions=None):
        super().__init__(
            "frequency",
            span,
            units="Hz",
            ureg=ureg,
            index=index,
            conditions=conditions,
            metadim="frequency",
        )


@type_registry.register
class LocusSet(LocusBase):

    def __init__(self, loci, **kwargs):
        super().__init__(**kwargs)
        self._loci = loci
        self._sort_loci()
        self.validate(loci)
       
    def __iter__(self):
        return iter(self.loci)

    def __len__(self):
        return len(self.loci)
    
    def __getitem__(self, idx):
        return self.loci[idx]

    @property
    def loci(self):
        return self._loci
    
    @property
    def all_conditions(self):
        return self.inherited_conditions | self.conditions
    
    @property
    def member_condition_names(self):
        return set().union(*(loc.conditions.keys() for loc in self.loci))
    
    def _sort_loci(self):
        pass

    def append(self, locus):
        self.validate([locus])
        self._loci.append(locus)
        self._sort_loci()

    def extend(self, loci):
        self.validate(loci)
        self._loci.extend(loci)
        self._sort_loci()

    def validate(self, loci):
        if self.loci and not all(type(locus) is type(self.loci[0]) for locus in loci):
            raise ValueError(
                f"All loci in a {self.__class__.__name__} must be of the same type."
            )

    def conditions_are_met(self, conditions):
        for key, value in conditions.items():
            if key in self.conditions and value != self.conditions[key]:
                return False
        return True

    def where(self, conditions=None, **kwargs):
        if conditions is None:
            conditions = {}
        conditions.update(kwargs)
        loci = [locus for locus in self.loci if locus.conditions_are_met(conditions)]
        if not len(loci):
            raise ValueError(
                f"Selection of {conditions} resulted in a length 0 {self.__class__.__name__}"
            )
        return self.filtered(loci)
    
    def filtered(self, loci):
        return type(self)(
            loci=loci,
            conditions=self.conditions,
            metadim=self.metadim,
            locus_type=self.locus_type
        )


@type_registry.register
class MarkerSet(LocusSet):
    marker_class = Marker

    def __init__(
        self, places=None, dim=None, markers=None, units=None, ureg=None, **kwargs
        ):
        if places is None and markers is None:
            raise ValueError("One of places or markers must not be None")
        if markers is None and dim is None:
            raise ValueError(
                "You must supply `dim` if you're not passing markers that already have one."
            )
        self.places = places
        self.units = units
        self.ureg = ureg
        self.dim = dim or markers[0].dim
        if markers is None:
            markers = self.markers_from_places()
        super().__init__(markers, **kwargs)
        self._loci = markers
        
       

    @property
    def markers(self):
        return self._loci

    def markers_from_places(self):
        markers = [
            self.marker_class(
                self.dim,
                place,
                units=self.units,
                index=i,
                ureg=self.ureg,
                conditions=self.conditions,
                metadim=self.metadim,
            )
            for i, place in enumerate(self.places)
        ]

        return markers

    def to_intervals(self, window):
        return self.to_interval_set(window)
    
    def _sort_loci(self):
        self._loci.sort(key=lambda interval: interval.span[0])

    def to_interval_set(self, window):
        intervals = [marker.to_interval(window) for marker in self.markers]
        return IntervalSet(
            self.dim,
            intervals=intervals,
            ureg=self.ureg,
            units=self.units,
            conditions=self.conditions,
            metadim=self.metadim,
        )
    
    def filtered(self, loci):
        return type(self)(
            markers=loci,
            dim=self.dim,
            conditions=self.conditions,
            units=self.units,
            ureg=self.ureg,
            metadim=self.metadim,
            locus_type=self.locus_type
        )


@type_registry.register
class EventSet(MarkerSet):
    def __init__(self, session, events=None, units="s", **kwargs):
        super().__init__(
            markers=events,
            dim="time",
            units=units,
            ureg=session.experiment.ureg,
            **kwargs
        )
        self.session = session
        self.metadim = "time"

    @property
    def events(self):
        return self.markers

    @property
    def to_epoch_set(self, window):
        return super().to_intervals(window)
    
    def _sort_loci(self):
        self._loci.sort(key=lambda event: event.value)
    
    def filtered(self, loci):
        return type(self)(
           self.session,
           events=loci,
           conditions=self.conditions,
           units=self.units,
           locus_type=self.locus_type
        )


@type_registry.register
class IntervalSet(LocusSet):
    locus_class = Interval

    def __init__(self, dim, spans=None, units=None, ureg=None, intervals=None, **kwargs):
        if spans is None and intervals is None:
            raise ValueError("One of intervals or spans must not be None")
        super().__init__(loci=intervals, **kwargs)
        self.dim = dim
        self.spans = spans
        self.units = units
        self.ureg = ureg
       
        if intervals is None:
            self.intervals_from_spans()

        self._sort_loci()

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
                metadim=self.metadim,
                locus_type=self.locus_type
            )
            for i, span in enumerate(self.spans)
        ]

    def filtered(self, loci):
        return type(self)(
            self.dim,
            intervals=loci,
            ureg=self.ureg,
            units=self.units,
            conditions=self.conditions,
            inherited_conditions=self.inherited_conditions,
            metadim=self.metadim,
            locus_type=self.locus_type
        )


@type_registry.register
class EpochSet(IntervalSet):
    locus_class = Epoch

    def __init__(self, session, spans=None, epochs=None, **kwargs):
        self.session = session
        super().__init__(
            "time", 
            spans=spans, 
            intervals=epochs, 
            ureg=session.experiment.ureg, 
            metadim="time", 
            **kwargs
        )

    def attrs_to_copy(self):
        attrs = super().attrs_to_copy()
        attrs.extend(["session"])
        return attrs

    @property
    def epochs(self):
        return self._loci
    
    def _sort_loci(self):
        self._loci.sort(key=lambda epoch: epoch.onset)

    def epochs_from_spans(self):
        self._loci = [
            Epoch(
                self.session,
                span[0],
                span[1] - span[0],
                units=self.units,
                index=i,
                conditions=self.conditions,
                inherited_conditions=self.inherited_conditions,
                locus_type=self.locus_type
            )
            for i, span in enumerate(self.spans)
        ]

    def intervals_from_spans(self):
        return self.epochs_from_spans()

    def validate_type(self, epochs):
        super().validate(epochs)
        if not all([isinstance(epoch, Epoch) for epoch in epochs]):
            raise ValueError("All members of an EpochSet must be of type Epoch.")
    
    def filtered(self, loci):
        return type(self)(
            self.session,
            epochs=loci,
            conditions=self.conditions,
            inherited_conditions=self.inherited_conditions,
            units=self.units,
            locus_type=self.locus_type
        )

       