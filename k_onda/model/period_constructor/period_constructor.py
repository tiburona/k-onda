from collections import defaultdict
import numpy as np


class PeriodConstructor:

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.target_periods = {}
        self.reference_periods = defaultdict(list)

    @property
    def earliest_period(self):
        return sorted([period for period in self.all_periods if not period.is_relative], 
                      key=lambda x: x.onset)[0]
    
    def get_all(self, attr):
        return [item for sublist in getattr(self, attr).values() for item in sublist]

    def select_children(self, attr):

        periods = self.get_all(attr)

        if self.selected_period_type:
            condition = lambda p: p.period_type == self.selected_period_type
    
        elif self.selected_period_types:
            condition = lambda p: p.period_type in self.selected_period_types
        elif self.selected_period_conditions:
            condition = lambda p: any(cond in self.selected_period_conditions 
                                         for cond in p.conditions)
        else:
            condition = lambda _: True

        if self.selected_period_group:
            prev_condition = condition
            def identifier(p):
                try:
                    return int(p.identifier)
                except:
                    return int(p.identifier.split('_')[-1])
            condition = lambda p: identifier(p) in self.selected_period_group and prev_condition(p)

        return [p for p in periods if condition(p)]


    def prepare_periods(self):
        self.period_class = self.experiment.kind_of_data_to_period_type[self.kind_of_data]
        try:
            period_info = self.period_info
        except AttributeError:
            period_info = self.animal.period_info

        periods = getattr(self, f"{self.kind_of_data}_periods")
        for is_relative, builder in (
            (False, self.construct_periods),
            (True, self.construct_relative_periods),
        ):
            for period_type, info in period_info.items():
                if bool(info.get('relative')) != is_relative:
                    continue
                periods[period_type] = builder(period_type, info)
        
    def construct_periods(self, period_type, period_info):
        periods = []
        if not period_info:
            return []
        # the time stamp of the beginning of a period
        period_onsets = period_info['onsets'] 
        
        period_events = self.event_times(period_info, period_onsets)
        
        for i, (onset, pe) in enumerate(zip(period_onsets, period_events)):
        
            periods.append(self.period_class(self, i, period_type, period_info, onset, 
                                             events=pe, experiment=self.experiment)) 
        return periods
    
    def event_times(self, period_info, period_onsets):
        events = period_info.get('events')
        if not events:
            return [[] for _ in period_onsets], None
        elif isinstance(events, dict):
            period_events = self.read_events_dict(events, period_onsets)   
        else:
            # the time stamps of things that happen within the period   
            period_events = period_info['events'] 

        return period_events
        
    def read_events_dict(self, events, period_onsets):
        # events is a dictionary like: {'start': 'period_onset', 'pattern': {'range_args': [30]}, 'unit': 'second',
        # 'event_times': []}
        # start can be 'period_onset' or a number indicating a position in the recording.  Default 0
        # if pattern exists, and is an iterable, that is a list of start times, or if it is a dictionary and
        # contains the key 'range_args', those are arguments that will give the start times when passed to the range
        # function
        # unit: 'second' or 'samples'.  Default 'second'.
        # if there is no pattern, then there must be a list of iterables with the start times
        # output is a list of lists of event times in seconds, counted from the beginning of the recording

        events_to_return = []
        unit = events.get('unit', 'second')
        start = events.get('start', 0)
        pattern = events.get('pattern')
        event_times = events.get('event_times', [])

        if isinstance(pattern, dict) and pattern.get('range_args'):
            pattern = list(range(*pattern['range_args']))

        for period_onset in period_onsets:

            events_list = np.array(pattern if pattern else event_times)
            
            if unit == 'second':
                events_list *= self.sampling_rate

            if start == 'period_onset':
                start_time = period_onset
            else:
                start_time = start if unit == 'sample' else start * self.sampling_rate

            events_list += start_time

            events_to_return.append(events_list)

        return events_to_return
   
    def construct_relative_periods(self, period_type, period_info):

        # period info for relative can optionally have a list, target_indices
        

        # if target_indices is not supplied, it's assumed that every target generates
        # a reference.  if target_indices is supplied (in the Rhonda case, target_index: 0)

        # relative periods are only constructed for those target indices

        # period info for non-relative periods can either not specify reference indices (old behavior)
        # supply one index, in which case all non-relative periods of this type are assumed to have
        # the same reference period, or a dictionary with keys target period indices 
        # and vals reference period indices.
        

        reference_periods = []
        modality_periods = getattr(self, f"{self.kind_of_data}_periods")
        candidate_target_periods = modality_periods[period_info['target']]
        target_index = period_info.get('target_index')
        reference_index = period_info.get('reference_index')
        exceptions = period_info.get('exceptions') 

        if target_index is None:
            target_index = list(range(len(candidate_target_periods)))
        elif isinstance(target_index, int):
            target_index = [target_index]
        else:
            pass

        for i in target_index:
            if exceptions and i in exceptions:
                shift = exceptions[i]['shift']
                duration = exceptions[i]['duration']
            else:
                shift = period_info['shift']
                duration = period_info.get('duration')
            shift_in_samples = shift * self.sampling_rate
            paired_period = candidate_target_periods[i]
            onset = paired_period.onset + shift_in_samples
            event_starts = []
           
            if period_info.get('events'):
                event_starts = self.event_times(period_info, [onset])[0]
            else:
                paired_event_starts = paired_period.event_starts
                for es in paired_event_starts:
                    ref_es = es + shift_in_samples
                    event_starts.append(ref_es)
            
            event_starts = np.array(event_starts)
            duration = duration if duration else paired_period.duration
            reference_period = self.period_class(self, i, period_type, period_info, onset, 
                                                 events=event_starts, target_period=paired_period, 
                                                 is_relative=True, experiment=self.experiment)
        
            reference_periods.append(reference_period)

        if reference_index is None:
            reference_index = {i:i for i in range(len(candidate_target_periods))}
        elif isinstance(reference_index, int):
            reference_index = {i:reference_index for i in range(len(candidate_target_periods))}
        else:
            pass

        for i, non_relative_period in enumerate(candidate_target_periods):
            non_relative_period.reference = reference_periods[reference_index[i]]
        return reference_periods