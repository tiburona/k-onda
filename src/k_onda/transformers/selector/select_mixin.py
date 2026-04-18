from dataclasses import dataclass, astuple, asdict

from k_onda.central import types
from k_onda.central import DimBounds, PadDimPair
from k_onda.utils import w_units
from k_onda.graph import list_nodes

from .selector import Selector, SelectionPlanner


@dataclass
class SelectParams:
    selection: str | object | None = None
    new_dim: str | None = None
    mode: str | None = None
    conditions: dict | None = None
    units: str | object | None = None
    window: tuple | list | None = None
    metadim: str | None = None
    kwargs: dict | None = None


class SelectMixin:
    def select(
            self, 
            selection=None, 
            new_dim=None, 
            mode=None, 
            conditions=None, 
            units=None, 
            window=None, 
            metadim=None,
            **kwargs
            ):
        
        # Although the underlying Transformer class has similar dispatch logic, `select` must handle 
        # its own dispatch because, in the case where a string like 'epochs' is passed as the 
        # selection, the resolution of the string into a Locus/LocusSet (like EpochSet) will be different 

        # TODO: I also have a class SignalMap?  is that currently being used?  Need to be added to type
        # dispatch here and in transformers
        
        params = SelectParams(
            selection, new_dim, mode, conditions, units, window, metadim, kwargs
        )
        
        if isinstance(self, types.Signal):
            return self.select_on_signal(signal=self, params=params)
        
        elif isinstance(self, types.Collection):
            return self.select_on_collection(collection=self, params=params) 
        
        elif isinstance(self, types.CollectionMap):
            group_on = getattr(self, "group_on", None)

            return types.CollectionMap(
                groups={
                    k: self.select_on_collection(v, params)
                    for k, v in self.items()
                },
                group_on=group_on,
            )

        elif isinstance(self, types.DataIdentity):
            return self.select_on_data_identity(data_identity=self, params=params)
        
    def select_on_collection(self, collection, params):
        d = asdict(params)
        extra = d.pop('kwargs') or {}
        
        return types.Collection([
            member.select(**d, **extra) for member in collection
        ])
    
    def select_on_data_identity(self, data_identity, params):
        return types.Collection(
            [self.select_on_signal(component, params) for component in data_identity.data_components]
        )

    def select_on_signal(self, signal, params):

        selection, new_dim, mode, conditions, units, window, metadim, kwargs = astuple(params)
            
        if isinstance(selection, str):
            
            session = signal.session if hasattr(signal, 'session') else signal.origin.session
            if selection == 'epochs':
                selection= session.epochs
            elif selection == 'events':
                selection = session.events
            else:
                raise ValueError(f"Unknown value {selection} passed to `select`.")
        
        if new_dim is not None and not isinstance(selection, (types.IntervalSet, types.MarkerSet)):
            raise ValueError("You can't create a new_dim unless you're selecting an " \
            "IntervalSet (or MarkerSet when it gets implemented)")
        
        dim_bounds = None

        if kwargs:
            dim_bounds, conditions = self.parse_select_kwargs(kwargs, selection, conditions)

        if dim_bounds:
            selection = self.locus_from_dim_bounds(signal, dim_bounds, units, conditions, params.metadim)

            
        if new_dim:
            if not metadim and not selection.metadim:
                raise ValueError("If you want to create a `new_dim` you must supply a `metadim`.")
            elif metadim and not selection.metadim:
                selection.metadim = metadim
            
        if window:
            if isinstance(selection, (types.Interval, types.IntervalSet)):
                raise ValueError("It doesn't make sense to define a Window on something that's" \
                "already an Interval or IntervalSet")
            else:
              
                window_span = PadDimPair(
                    w_units(
                        window, 
                        selection.metadim, 
                        units, 
                        signal.origin.session.experiment.ureg
                    ))
                
                selection = selection.to_intervals(window_span)
                window = DimBounds({selection.metadim: window_span})
            
        return Selector(mode, selection, new_dim, window)(signal)

    def parse_select_kwargs(self, kwargs, selection, conditions):

        dim_bounds = {}

        # if either conditions or selection have been explicitly passed to `select`,
        # assume all kwargs belong to the other.
        if conditions:
            dim_bounds = kwargs
        elif selection:
            conditions = kwargs
        else:
            dim_bounds = {k: v for k, v in kwargs.items if self.is_inferrably_a_dim(k)}
            conditions = {k: v for k, v in kwargs.items if not self.is_inferrably_a_dim(k)}
        
        return dim_bounds, conditions

    def is_inferrably_a_dim(self, key):
        coord_contains_patterns = ['time', 'freq', 'spike', 'sample', 'loc', 'pos']
        coord_is_patterns = ['x', 'y']
        if any([string in key for string in coord_contains_patterns]):
            return True
        if any([string == key for string in coord_is_patterns]):
            return True
        else:
            return False
        
    def filter_selection_by_conditions(self, selection, conditions):
        return selection.where(**conditions)

    def locus_from_dim_bounds(self, signal, dim_bounds, units, conditions, metadim):

        dim = list(dim_bounds.keys())[0]
        span = list(dim_bounds.values())[0]

        ureg=signal.origin.session.ureg

        cls = types.Interval if isinstance(dim_bounds, dict) else types.IntervalSet
            
        return cls(dim, span, ureg=ureg, units=units, metadim=metadim, conditions=conditions)  

    def plan_selection(self):

        if isinstance(self, types.Signal):
            return self.plan_on_signal(self)
        elif isinstance(self, types.Collection):
            return self.plan_on_collection(self)
        elif isinstance(self, types.CollectionMap):
            group_on = getattr(self, "group_on", None)

            return types.CollectionMap(
                groups={
                    k: self.plan_on_collection(v)
                    for k, v in self.items()
                },
                group_on=group_on,
            )
            
    def plan_on_signal(self, signal):
        if signal._selection_planned:
            return
        for node in list_nodes(signal):
            node._selection_planned = True
        SelectionPlanner()(signal)

    def plan_on_collection(self, collection):
        return types.Collection([self.plan_on_signal(signal) for signal in collection.members])

