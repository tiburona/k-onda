
import numpy as np
from collections import defaultdict
from bisect import bisect_left as bs_left, bisect_right as bs_right
from base_data import Data
from period_event import Period, Event
from period_constructor import PeriodConstructor
from math_functions import calc_rates, calc_hist, cross_correlation, correlogram
from bins import BinMethods
from phy_interface import PhyInterface
from prep_methods import PrepMethods
import os
import h5py
from utils import group_to_dict


class SpikeMethods:

    def get_psth(self):
        return self.get_average('get_psth', stop_at=self.calc_opts.get('base', 'event'))
    
    def get_firing_rates(self):
        return self.get_average('get_firing_rates', stop_at=self.calc_opts.get('base', 'event'))
    
    def get_spike_counts(self):
        return self.get_average('get_spike_counts', stop_at=self.calc_opts.get('base', 'event'))
    
    
    
class RateMethods:

    # Methods shared by SpikePeriod and SpikeEvent
    
    @property
    def spikes(self):
        return [spike for spike in self.unit.find_spikes(*self.spike_range)]
    
    @property
    def spikes_in_seconds_from_start(self):
        return [spike - self.start for spike in self.unit.find_spikes(*self.spike_range)]
    
    @property
    def spike_range(self):
        return (self.start, self.stop)
    
    def get_psth(self):
        return self._get_calc('psth')
    
    def get_firing_rates(self):
        return self._get_calc('firing_rates')
    
    def get_spike_counts(self):
        return self._get_calc('spike_counts')
    
    def get_spike_train(self):
        return self._get_calc('spike_train')
        
    def _get_calc(self, calc_type):
        stop_at=self.calc_opts.get('base', 'event')
        if self.name == stop_at:
            return getattr(self, f"_get_{calc_type}")()
        else:
            return self.get_average(f"get_{calc_type}", stop_at=stop_at)
        
    def _get_spike_counts(self):
        if self.unit.category == 'mua':
            a = 'foo'
        bin_size = self.calc_opts.get('bin_size', .01)
        if 'counts' in self.private_cache:
            counts = self.private_cache['counts']
        else:
            counts = calc_hist(self.spikes, self.num_bins_per, self.spike_range)[0]
        if self.calc_type == 'psth':
            self.private_cache['counts'] = counts
        return self.refer(counts)
    
    def _get_spike_train(self):
        return np.where(self._get_spike_counts() != 0, 1, 0)
        
    def _get_psth(self):
        rates = self.get_firing_rates() 
        reference_rates = self.reference.get_firing_rates()
        rates -= reference_rates
        rates /= self.unit.get_firing_std_dev(period_types=self.period_type,)  # same as dividing unit psth by std dev 
        self.private_cache = {}
        return rates

    def _get_firing_rates(self):
        return self._get_firing_counts()/self.calc_opts.get('bin_size', .01)


class Unit(Data, PeriodConstructor, SpikeMethods):

    _name = 'unit'
    
    def __init__(self, animal, category, spike_times, cluster_id, waveform=None, experiment=None, 
                 neuron_type=None, quality=None):
        super().__init__()
        PeriodConstructor().__init__()
        self.animal = animal
        self.category = category
        self.spike_times = np.array(spike_times)
        self.cluster_id = cluster_id
        self.waveform = waveform
        self.experiment = experiment
        self.neuron_type = neuron_type
        self.quality = quality
        self.animal.units[category].append(self)
        self.identifier = str(self.animal.units[category].index(self) + 1)
        self.spike_periods = defaultdict(list)
        self.parent = animal
        self.kind_of_data_to_period_type = {
            'spike': SpikePeriod
        }

    @property
    def children(self):
        return self.select_children('spike_periods')
        
    @property
    def all_periods(self):
        return [period for key in self.spike_periods for period in self.spike_periods[key]]

    @property
    def firing_rate(self):
        return len(self.spike_times) / (float(self.spike_times[-1] - self.spike_times[0]))

    @property
    def unit_pairs(self):
        all_unit_pairs = self.get_pairs()
        pairs_to_select = self.calc_opts.get('unit_pair')
        if pairs_to_select is None:
            return all_unit_pairs
        else:
            return [unit_pair for unit_pair in all_unit_pairs if ','.join(
                [unit_pair.unit.neuron_type, unit_pair.pair.neuron_type]) == pairs_to_select]
 
    def spike_prep(self):
        self.prepare_periods()

    def get_pairs(self):
        return [UnitPair(self, other) for other in [unit for unit in self.animal if unit.identifier != self.identifier]]

    def find_spikes(self, start, stop):
        return np.array(self.spike_times[bs_left(self.spike_times, start): bs_right(self.spike_times, stop)])

    def get_spikes_by_events(self):
        return [event.spikes for period in self.children for event in period.children]

    def get_firing_std_dev(self):
        return np.std([self.concatenate(method='get_firing_rates', level=-2)])

    def get_cross_correlations(self, axis=0):
        return np.mean([pair.get_cross_correlations(axis=axis, stop_at=self.calc_opts.get('base', 'period'))
                        for pair in self.unit_pairs], axis=axis)

    def get_correlogram(self, axis=0):
        return np.mean([pair.get_correlogram(axis=axis, stop_at=self.calc_opts.get('base', 'period'))
                        for pair in self.unit_pairs], axis=axis)
    
    def get_waveform(self):
        if self.waveform is not None:
            return self.waveform
        else:
            phy = PhyInterface(self.calc_opts['data_path'], self.parent.identifier)
            electrodes = phy.cluster_dict[self.cluster_id]['electrodes']
            wf = phy.get_mean_waveforms(self.cluster_id, electrodes)
            self.waveform = wf
            return wf
        
    def get_raster(self):
        base = self.calc_opts.get('base', 'event')
        # raster type can be spike_train for binarized data or spike_counts for gradations
        raster_type = self.calc_opts.get('raster_type', 'spike_train')
        depth = 1 if base == 'period' else 2
        raster = self.get_stack(method=f"get_{raster_type}", depth=depth)
        
        return raster
        
        

class UnitPair:
    pass



class SpikePeriod(Period, RateMethods):

    name = 'period'

    def __init__(self, unit, index, period_type, period_info, onset, 
                 events=None, target_period=None, is_relative=False, 
                 experiment=None):
        super().__init__(index, period_type, period_info, onset, events=events, 
                         experiment=experiment, target_period=target_period, 
                         is_relative=is_relative)
        self.unit = unit
        self.animal = self.unit.animal
        self.parent = unit
        self.private_cache = {}
        self._start = self.onset/self.sampling_rate 
        self._stop = self.start + self.duration 
        
    @property
    def start(self):
        return self._start - self.pre_period
    
    @property
    def stop(self):
        return self._stop + self.post_period
        
        
    def get_events(self):
        self._events = [SpikeEvent(self, self.unit, start, i) 
                        for i, start in enumerate(self.event_starts)]


class SpikeEvent(Event, RateMethods, BinMethods):
    def __init__(self, period, unit, start,  index):
        super().__init__(period, index)
        self.unit = unit
        self.private_cache = {}
        self._start = start

    @property
    def start(self):
        self._start - self.pre_event

    @property
    def stop(self):
        self._start + self.post_event

    @property
    def spike_range(self):
        return (-self.pre_event, self.post_event)
    
    def get_spike_counts(self):
        return calc_hist(self.spikes, self.num_bins_per, self.spike_range)

    def get_cross_correlations(self, pair=None):
        other = pair.periods[self.period_type][self.period.identifier].events[self.identifier]
        cross_corr = cross_correlation(self.get_unadjusted_rates(), other.get_unadjusted_rates(), mode='full')
        boundary = round(self.calc_opts['max_lag'] / self.calc_opts['bin_size'])
        midpoint = cross_corr.size // 2
        return cross_corr[midpoint - boundary:midpoint + boundary + 1]

    def get_correlogram(self, pair=None, num_pairs=None):
        max_lag, bin_size = (self.calc_opts[opt] for opt in ['max_lag', 'bin_size'])
        lags = round(max_lag/bin_size)
        return correlogram(lags, bin_size, self.spikes, pair.spikes, num_pairs)

    def get_autocorrelogram(self):
        max_lag, bin_size = (self.calc_opts[opt] for opt in ['max_lag', 'bin_size'])
        lags = round(max_lag / bin_size)
        return correlogram(lags, bin_size, self.spikes, self.spikes, 1)
    

class SpikePrepMethods(PrepMethods):

    def spike_prep(self):
        self.unit_prep()
        if 'periods_from_nev' in self.period_info.get('instructions', []):
            self.period_info.update(self.get_periods_from_nev())
        for unit in [unit for category, units in self.units.items() for unit in units]:
            unit.spike_prep()
            # if self.selected_period_type:
            #     unit.children = unit.spike_periods[self.selected_period_type]
            # else:
            #     unit.children = [period for pt, periods in unit.spike_periods.items() 
            #                      for period in periods]
        
    def select_spike_children(self):
        if self.selected_neuron_type:
            return getattr(self, self.selected_neuron_type)
        else: 
            return self.units['good']

    def get_periods_from_nev(self):
        file_path = self.animal_info.get('nev_file_path')
        if not file_path:
            file_path = self.construct_path('nev')
        periods_with_code = {k: v for k, v in self.animal_info['period_info'].items() if 'code' in v}

        with h5py.File(file_path, 'r') as mat_file:
            data = group_to_dict(mat_file['NEV'])
            for period_type, period_dict in periods_with_code.items():
                onsets = []
                for i, code in enumerate(data['Data']['SerialDigitalIO']['UnparsedData'][0]):
                    if code == period_dict['code']:
                        onsets.append(int(data['Data']['SerialDigitalIO']['TimeStamp'][i][0]))
                periods_with_code[period_type]['onsets'] = onsets
        return periods_with_code
    
    def unit_prep(self):
        if any(d.get('get_units_from_phy') for d in [self.animal_info, self.experiment.exp_info]):
            
            phy_interface = PhyInterface(self.construct_path('phy'), self)
            cluster_dict = phy_interface.cluster_dict

            for cluster, info in cluster_dict.items():
                if info['group'] != 'noise':
                    try:
                        waveform = phy_interface.get_mean_waveforms_on_peak_electrodes(cluster)
                    except ValueError as e:
                        # Kilosort2 apparently sometimes creates a cluster with no spikes that phy
                        # doesn't display, so you don't mark it noise
                        if "argmax of an empty sequence" in str(e):
                            continue
                        else:
                            raise
                    waveform = phy_interface.get_mean_waveforms_on_peak_electrodes(cluster)
                    _ = Unit(self, info['group'], info['spike_times'], cluster, waveform=waveform, 
                             experiment=self.experiment)
        else:
            pass
                   

    

    
