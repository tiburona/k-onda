import numpy as np


class Marker:

    def __init__(self, dim, value, index=None, units=None, conditions=None):
        self.dim = dim
        self.value = value
        self.index = index
        self.units = units
        self.conditions = conditions

    def to_interval(self, window):
        return Interval(self.dim, window, self.units, self.index, self.conditions)


class Event(Marker):

    def __init__(self, value, index=None, units=None, conditions=None):
        super().__init__('time', value, index, units, conditions)


class Interval:

    marker_class = Marker

    def __init__(self, dim, span, units=None, index=None, conditions=None):
        self.dim = dim
        self.span = span # tuple of lo, hi
        self.units = units
        self.index = index
        self.conditions = conditions
        self.extent = self.span[1] - self.span[0]

    def generate_markers(self, spacing=None, offsets=None, count=None, positions=None, linspace=None):
        if len([arg for arg in (spacing, offsets, count, positions, linspace) is not None]) != 1:
            raise ValueError("You must pass exactly 1 keyword argument to generate_markers")
        if count is not None:
            markers = np.arange(count) * (self.extent)/count + self.span[0]
        elif spacing is not None:
            markers = np.arange(self.span[0], self.span[1], spacing)
        elif offsets is not None:
            markers = [offset + self.span[0] for offset in offsets]
        elif positions is not None:
            markers = [position for position in positions if self.span[0] <= position < self.span[1]]
        elif linspace is not None:
            markers = np.linspace(self.span[0], self.span[1], linspace)

        return [
            self.marker_class(self.dim, marker, index=i, units=self.units, conditions=self.conditions) 
            for i, marker in enumerate(markers)
            ]
            



class Epoch(Interval):

    marker_class = Event
    
    def __init__(self, session, onset, duration, units='s', index=None, conditions=None):
        self.session = session
        self.onset = onset
        self.duration = duration
        self.t0 = self.onset
        self.t1 = self.onset + self.duration
        self.span = (self.t0, self.t1)
       
        super().__init__(
            'time', 
            self.span, 
            units=units, 
            index=index, 
            conditions=conditions)
         
    def generate_events(
            self, 
            spacing=None, 
            offsets=None, 
            count=None, 
            positions=None, 
            linspace=None
            ):
        return self.generate_markers(spacing, offsets, count, positions, linspace)
     

class FrequencyBand(Interval):

    def __init__(self, span, index=None, conditions=None):
        super().__init__('frequency', span, 'Hz', index=index, conditions=conditions)



class LocusSet:
    # TODO put set algebra here
    pass


class MarkerSet(LocusSet):

    def __init__(self, markers, conditions):
        self.markers = markers
        self.conditions = conditions
    
    def to_interval_set(self, window):
        intervals = [marker.to_interval(window) for marker in self.markers]
        return IntervalSet(intervals, self.conditions)


class IntervalSet(LocusSet):

    def __init__(self, intervals, conditions):
        self.intervals = intervals
        self.conditions = conditions

