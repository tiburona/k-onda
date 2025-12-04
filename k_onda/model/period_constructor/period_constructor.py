from collections import defaultdict
import numpy as np
import xarray as xr
import pint
import pint_xarray


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
        
        def make_unit_aware(onset):
            if self.period_info.get('1_indexed'):
                onset -= 1
            return self.quantity(
                value=onset,
                units="raw_sample", 
                name="onset"
            )

        # the time stamp of the beginning of a period
        period_onsets = [make_unit_aware(onset) for onset in period_info['onsets']]
        
        period_events = self.event_times(period_info, period_onsets)

        duration = self.quantity(period_info['duration'], units='second', 
                                 name='duration')
        
        for i, (onset, pe) in enumerate(zip(period_onsets, period_events)):
        
            periods.append(
                self.period_class(
                    self, i, period_type, period_info, onset, events=pe, 
                    duration=duration, experiment=self.experiment)
                    ) 
            
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
        """
        events is a dictionary like:
            {
                'start': 'period_onset',
                'pattern': {'range_args': [30]},
                'unit': 'second',
                'event_times': []
            }

        start: 'period_onset' or a numeric offset in the given unit (default 0).
        pattern:
        - if a dict with 'range_args', we call range(*range_args)
        - otherwise, it can be an iterable of offsets
        unit: 'second' or 'raw_sample' (default 'second').
        event_times: list-of-lists of offsets per period if pattern is not given.

        Returns:
            list of lists of event times in seconds, absolute from recording start.
        """

        events_to_return = []
        unit = events.get("unit", "second")
        start_cfg = events.get("start", 0)
        pattern = events.get("pattern")
        event_times = events.get("event_times", [])

        # Expand pattern if given as range_args
        if isinstance(pattern, dict) and pattern.get("range_args"):
            pattern = list(range(*pattern["range_args"]))

        for i, period_onset in enumerate(period_onsets):
            # Choose the relative times for this period
            if pattern is not None:
                rel_times = pattern
            else:
                rel_times = event_times[i]

            # Wrap relative times as quantities in the requested unit
            rel_times_q = [
                self.quantity(event, units=unit, name="event_offset")
                for event in rel_times
            ]

            # Determine the base start for this period
            if start_cfg == "period_onset":
                start_q = period_onset  # assume already a quantity
            else:
                start_q = self.quantity(start_cfg, units=unit, name="event_start_base")

            # Absolute times = start + offset
            abs_times = [
                (offset + start_q)
                for offset in rel_times_q
            ]

            events_to_return.append(abs_times)

        return events_to_return

    def construct_relative_periods(self, period_type, period_info):

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

        for i in target_index:

            paired_period = candidate_target_periods[i]

            if exceptions and i in exceptions:
                shift_val = exceptions[i]['shift']
                duration_val = exceptions[i]['duration']
            else:
                shift_val = period_info['shift']
                duration_val = period_info.get('duration')

            shift = self.quantity(shift_val, units='second', name="shift")

            if duration_val is not None:
                duration = self.quantity(duration_val, units='second', name='duration')
            else:
                duration = paired_period.duration
            
            onset = paired_period.onset + shift   # result in raw_sample

            if period_info.get('events'):
                event_starts = self.event_times(period_info, [onset])[0]
            else:
                event_starts = [es + shift for es in paired_period.event_starts]

            reference_period = self.period_class(
                self,
                i,
                period_type,
                period_info,
                onset,
                events=event_starts,
                duration=duration,
                shift=shift,
                target_period=paired_period,
                is_relative=True,
                experiment=self.experiment,
            )

            reference_periods.append(reference_period)

        # reference_index logic really wants to be consistent with target_index
        if reference_index is None:
            reference_index = {i: i for i in target_index}
        elif isinstance(reference_index, int):
            reference_index = {i: reference_index for i in target_index}

        for i in target_index:
            non_relative_period = candidate_target_periods[i]
            non_relative_period.reference = reference_periods[reference_index[i]]

        return reference_periods
   
    def construct_relative_periods(self, period_type, period_info):        

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

            shift = self.quantity(
                shift,
                units='second',
                name = "shift"
                )
            
            if duration is not None:
                duration = self.quantity(
                    duration,
                    units='second',
                    name = 'duration'
                )

            paired_period = candidate_target_periods[i]
            onset = paired_period.onset + shift

            event_starts = []
           
            if period_info.get('events'):
                event_starts = self.event_times(period_info, [onset])[0]
            else:
                paired_event_starts = paired_period.event_starts
                for es in paired_event_starts:
                    ref_es = es + shift
                    event_starts.append(ref_es)
            
            event_starts = np.array(event_starts)
            duration = duration if duration else paired_period.duration
            reference_period = self.period_class(self, i, period_type, period_info, onset, 
                                                 events=event_starts, target_period=paired_period, 
                                                 is_relative=True, experiment=self.experiment)
        
            reference_periods.append(reference_period)

        # target_index is a list like [2, 3, 4]
        # reference_index wants to be a mapping of the index of a target period ->
        # the index of a reference period

        if reference_index is None:
            if target_index != list(range(len(candidate_target_periods))):
                raise ValueError(
                "If target_index is a subset of target periods, " \
                "you must explicitly provide reference_index."
                )
            # assume every target period uses the reference period whose creation
            # it just triggered
            reference_index = {i: i for i in target_index} 

        elif isinstance(reference_index, int):
            # every target period uses the same reference period
            reference_index = {i: reference_index 
                               for i in range(len(candidate_target_periods))}

        for i in target_index:
            non_relative_period = candidate_target_periods[i]
            non_relative_period.reference = reference_periods[reference_index[i]]