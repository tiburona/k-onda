import numpy as np


class PeriodConstructor:

    @property
    def all_periods(self):
        return [period for period_type, period_list in self.periods.items() for period in period_list]

    @property
    def earliest_period(self):
        return sorted([period for period in self.all_periods if not period.is_relative], key=lambda x: x.onset)[0]

    def prepare_periods(self):
        for boo, function in zip((False, True), (self.construct_periods, self.construct_relative_periods)):
            try:
                period_info = self.period_info
            except AttributeError:
                period_info = self.parent.period_info
            filtered_period_info = {k: v for k, v in period_info.items() if bool(v.get('relative')) == bool(boo)}
            for period_type in filtered_period_info:
                self.periods[period_type] = function(period_type, filtered_period_info[period_type])

    def construct_periods(self, period_type, period_info):
        periods = []
        num_events = len([event for events_list in period_info['events'] for event in
                          events_list])  # all the events for this period type

        if self.data_opts and self.data_opts.get('events', {}).get(period_type, {}).get('selection') is not None:
            events = slice(*self.data_opts['events'][period_type]['selection'])
        else:
            events = slice(0, num_events)  # default is to take all events
        selected_event_indices = list(range(num_events))[events]  # indices of the events used in this data analysis
        period_onsets = period_info['onsets']  # the time stamp of the beginning of a period
        period_events = period_info['events']  # the time stamps of things that happen within the period
        event_ind = 0
        for i, (onset, events) in enumerate(zip(period_onsets, period_events)):
            period_events = np.array([ev for j, ev in enumerate(events) if event_ind + j in selected_event_indices])
            if self.name == 'animal': # self is an LFPAnimal
                onset = onset * self.sampling_rate/self.spike_target.sampling_rate
                period_events = period_events * self.sampling_rate/self.spike_target.sampling_rate
            event_ind += len(period_info['events'][i])
            periods.append(self.period_class(self, i, period_type, period_info, onset, events=period_events))
        return periods

    def construct_relative_periods(self, period_type, period_info):
        periods = []
        shift = period_info['shift']
        duration = period_info.get('duration')
        paired_periods = self.periods[period_info['target']]

        for i, paired_period in enumerate(paired_periods):
            if self.name == 'animal':  # if self is animal this is an lfp period
                shift += sum(paired_period.convolution_padding)
            onset = paired_period.onset - shift * self.sampling_rate
            event_starts = paired_period.event_starts - shift * self.sampling_rate
            duration = duration if duration else paired_period.duration
            reference_period = self.period_class(self, i, period_type, period_info, onset, events=event_starts,
                                               target_period=paired_period, is_relative=True)
            paired_period.paired_period = reference_period
            periods.append(reference_period)
        return periods

    def filter_by_selected_periods(self, periods):
        # Difference between self.selected_period_type and self.data_opts.get('periods'): the former comes from the
        # context, and is set internally to the program by a plotter that iterates over period types.  The latter comes
        # from the user's analysis config, and is used to restrict a plot to a subset of periods

        if self.selected_period_type is not None:
            children = periods[self.selected_period_type]
        else:
            children = [period for period_type in periods for period in periods[period_type]]
        if self.frozen_periods is not None:
            selected_periods = self.frozen_periods
        else:
            selected_periods = self.data_opts.get('periods')
        if selected_periods is None:
            return children
        else:
            return [child for child in children if self.is_selected(child, selected_periods)]

    def is_selected(self, x, selected_periods):
        child_id = lambda x: x.period.identifier if self.data_type == 'mrl' else x.identifier
        return x.period_type not in selected_periods or child_id(x) in selected_periods[x.period_type]
