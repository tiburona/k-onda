
import numpy as np
from collections import defaultdict
from bisect import bisect_left as bs_left, bisect_right as bs_right
from neo.rawio import BlackrockRawIO

import matplotlib.pyplot as plt
plt.ioff() 

from collections import defaultdict
from utils import save, load
from phy_interface import PhyInterface
from base_data import Data, Base
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
        return [spike - self.start 
                for spike in self.unit.find_spikes(*self.spike_range)]
    
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
        rates /= self.unit.get_firing_std_dev()  # same as dividing unit psth by std dev 
        self.private_cache = {}
        return rates

    def _get_firing_rates(self):
        return self._get_spike_counts()/self.calc_opts.get('bin_size', .01)


class Unit(Data, PeriodConstructor, SpikeMethods):

    _name = 'unit'
    
    def __init__(self, animal, category, spike_times, cluster_id, waveform=None, experiment=None, 
                 neuron_type=None, quality=None, firing_rate=None, fwhm_seconds=None):
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
        self.firing_rate = firing_rate
        self.fwhm_seconds = fwhm_seconds
        self.fwhm_microseconds = fwhm_seconds * 1e6 if fwhm_seconds else None
        self.animal.units[category].append(self)
        self.identifier = '_'.join([self.animal.identifier, self.category, 
                                    str(self.animal.units[category].index(self) + 1)])
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
        depth = 2 if self.has_grandchildren else 1
        return np.std([self.concatenate(method='get_firing_rates', depth=depth)])

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
        self._stop = self._start + self.duration 
        
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
    def __init__(self, period, unit, onset,  index):
        super().__init__(period, index)
        self.unit = unit
        self.private_cache = {}
        self._start = onset/self.sampling_rate

    @property
    def start(self):
       return self._start - self.pre_event

    @property
    def stop(self):
        return self._start + self.post_event

    @property
    def spike_range(self):
        return (self.start, self.stop)
    
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
        self.units_prep()
        for unit in [unit for _, units in self.units.items() for unit in units]:
            unit.spike_prep()

            # if self.selected_period_type:
            #     unit.children = unit.spike_periods[self.selected_period_type]
            # else:
            #     unit.children = [period for pt, periods in unit.spike_periods.items() 
            #                      for period in periods]
        
    def select_spike_children(self):
        if self.selected_neuron_type:
            return self.neurons[self.selected_neuron_type]
        else: 
            return [unit for units in self.units.values() for unit in units]
        
    def units_prep(self):
        units_info = self.animal_info.get('units', {})
        if 'get_units_from_phy' in units_info.get('instructions', []):
            self.get_units_from_phy()

        for category in [cat for cat in ['good', 'MUA'] if cat in units_info]:
            for unit_info in units_info[category]:
                unit_kwargs = {kw: unit_info[kw] for kw in ['waveform', 'neuron_type', 'quality', 'firing_rate', 'fwhm_seconds']}
                unit = Unit(self, category, unit_info['spike_times'], unit_info['cluster'], 
                            experiment=self.experiment, **unit_kwargs)
                if unit.neuron_type:
                    getattr(self, unit.neuron_type).append(unit)

    def get_units_from_phy(self):
        saved_calc_exists, units, pickle_path = load(
            os.path.join(self.construct_path('dest'), 'units_from_phy.pkl'), 'pkl'
        )
        
        if saved_calc_exists:
            for unit in units:
                Unit(self, unit['group'], unit['spike_times'], unit['cluster'], 
                    waveform=unit['waveform'], experiment=self.experiment, 
                    firing_rate=unit['firing_rate'], fwhm_seconds=unit['fwhm_seconds'])
        else:
            phy_path = self.construct_path('phy')
            phy_interface = PhyInterface(phy_path, self)
            cluster_dict = phy_interface.cluster_dict
            units = []

            for cluster, info in cluster_dict.items():
                if info['group'] in ['good', 'mua']:
                    try:
                        waveform = phy_interface.get_mean_waveforms_on_peak_electrodes(cluster)
                    except ValueError as e:
                        if "argmax of an empty sequence" in str(e):
                            continue
                        else:
                            raise
                    
                    # Check if deflection is up or down at the center
                    center_value = waveform[100]
                    surrounding_mean = np.mean(np.concatenate((waveform[0:50], waveform[-50:])))
                    threshold = (np.max(waveform[70:130]) + np.min(waveform[70:130])) / 2

                    if center_value > surrounding_mean:  # Upward deflection
                        above_threshold = waveform >= threshold
                    else:  # Downward deflection
                        above_threshold = waveform <= threshold

                    # Find transitions (crossings)
                    transitions = np.diff(above_threshold.astype(int))

                    # Last False -> True transition before center (rising edge)
                    before_center = np.where(transitions[:101] == 1)[0]
                    start = before_center[-1] if len(before_center) > 0 else 70  # fallback start

                    # First True -> False transition after center (falling edge)
                    after_center = np.where(transitions[100:] == -1)[0] + 100
                    end = after_center[0] if len(after_center) > 0 else 130  # fallback end

                    # Calculate FWHM in samples and time
                    FWHM_samples = end - start
                    FWHM_seconds = FWHM_samples / self.sampling_rate

                    # Plot waveform and mark FWHM
                    plt.figure(figsize=(10, 5))
                    plt.plot(waveform, label="Waveform")
                    
                    # Horizontal line for half maximum level
                    plt.axhline(y=threshold, color='r', linestyle='--', label="Half Maximum")

                    # Mark FWHM region
                    plt.axvspan(start, end, color='orange', alpha=0.3, label="FWHM Range")
                    
                    plt.xlabel("Sample Index")
                    plt.ylabel("Amplitude")
                    plt.title(f"Waveform with FWHM for Animal {self.identifier} Cluster {cluster}")
                    plt.legend()
                    plt.savefig(os.path.join(phy_path, f"{self.identifier}_{cluster}_waveform.png"))
                    
                    # Firing rate calculation
                    spike_times = np.array(info['spike_times'])
                    spike_times_for_fr = spike_times[spike_times > 30]
                    firing_rate = len(spike_times_for_fr) / (spike_times_for_fr[-1] - spike_times_for_fr[0])

                    # Create unit and add attributes
                    unit = Unit(self, info['group'], info['spike_times'], cluster,
                                waveform=waveform, experiment=self.experiment, firing_rate=firing_rate, 
                                fwhm_seconds=FWHM_seconds)

                    # Append to units list
                    units.append(info | {'waveform': waveform, 'firing_rate': firing_rate, 'fwhm_seconds': FWHM_seconds})
            
            save(units, pickle_path, 'pkl')




            


            
                