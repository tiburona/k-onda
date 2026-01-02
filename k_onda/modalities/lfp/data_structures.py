from collections import OrderedDict
import numpy as np
import xarray as xr
from mne.time_frequency import tfr_array_multitaper
from copy import deepcopy
import json
from scipy.signal.windows import tukey

from k_onda.math import apply_hilbert_to_padded_data, welch_psd
from k_onda.model.period_event import Period, Event
from k_onda.model import Data
from k_onda.model.bins import TimeBin
from k_onda.interfaces import MatlabInterface
from k_onda.math import normalized_xcorr
from ...modalities.mixins import BandPassFilterMixin
from k_onda.utils import (is_iterable, contains_nan)
from .methods import LFPMethods, SpectralDensityMethods, CoherenceMethods
from .data_structures_mixins.event_validator import EventValidation
from .data_structures_mixins.descendant_cache import DescendantCache


class LFPProperties:

    @property
    def tolerance(self):
        tolerance = self.calc_opts.get('frequency_tolerance', .2)
        return self.quantity(tolerance, units='Hz', name='tol')

    @property
    def lfp_padding(self):
        padding = self.calc_opts.get('lfp_padding', [0, 0])

        padding = self.quantity(
            padding,
            units='second',
            dims=('side',),
            coords={'side': ['pre', 'post']},
            name='lfp_padding'
        )
        return padding.pint.to('lfp_sample')
        
    @property
    def pad_start(self):
        start = (self.start - self.lfp_padding.sel(side='pre')).pint.to('lfp_sample')
        return start

    @property
    def pad_stop(self):
        return (self.stop + self.lfp_padding.sel(side='post')).pint.to('lfp_sample')
    
    @property
    def lost_signal(self):
        lost_signal = self.calc_opts.get('lost_signal', [0, 0])
        return self.quantity(
            lost_signal,
            units='second',
            dims=('side',),
            coords={'side': ['pre', 'post']},
            name='lost_signal'
        )
    

class LFPPeriod(LFPMethods, Period, LFPProperties, EventValidation, SpectralDensityMethods, 
                DescendantCache):

    def __init__(self, animal, index, period_type, period_info, onset, duration, events=None,
                  target_period=None, is_relative=False, experiment=None):
        super().__init__(index, period_type, period_info, onset, duration, experiment=experiment, 
                         target_period=target_period, is_relative=is_relative, events=events,
                         )
        self.animal = animal
        self.parent = animal
        self.event_starts = events    
        self._spectrogram = None
        self._segments = OrderedDict()
        self._events = OrderedDict()
        self._valid_events = OrderedDict()
        self.max_event_cache   = self.calc_opts.get('max_event_cache', 4)
        self.max_segment_cache = self.calc_opts.get('max_segment_cache', 4)

    @property
    def events(self):
        # Use the descendant cache so we return the cached list, not the cache mapping
        return self._cached_collection(
            cache_name="_events",
            key_fn=self._event_key,
            build_fn=self.get_events,
            max_size=self.max_event_cache,
        )
    
    @property
    def padded_data(self):
        return self.get_data_from_animal_dict(pad=True)
    
    @property
    def unpadded_data(self):
        return self.get_data_from_animal_dict(pad=False)

    @property
    def event_starts_in_period_time(self):
        if getattr(self, '_event_starts_in_period_time', None) is None:
            self._event_starts_in_period_time = xr.concat(
                [(t_abs - self.onset).pint.to('second') for t_abs in self.event_starts],
                dim="event"
            )
        return self._event_starts_in_period_time
    
    def get_data_from_animal_dict(self, pad=False):
        data_source = self.animal.processed_lfp
        idx_qs = (self.pad_start, self.pad_stop) if pad else (self.start, self.stop)
        start, stop = [self.to_int(idx_q, 'lfp_sample') for idx_q in idx_qs]

        if self.selected_brain_region:
            return data_source[self.selected_brain_region][start:stop]
        else:
            return {brain_region: data_source[brain_region][start:stop] 
                    for brain_region in data_source}
    
    @property
    def spectrogram(self):
        if self._spectrogram is None:
            self._spectrogram = self.calc_spectrogram()
        # todo: should cache the frequency selected spectrogram
        return self.frequency_selector(self._spectrogram)
    
    @property
    def children(self):
        if (self.calc_type in ['coherence', 'psd'] 
            and self.calc_opts.get('validate_events') == True
            and self.calc_opts.get('coherence_params', {}).get('method') != 'multitaper'):
            return self.get_segments()
        else:
            if not self.calc_opts.get('validate_events'):
                return self.events
            return self.get_valid_events()
        
    def get_events(self):

        if self.pre_event + self.post_event == 0:
            raise ValueError("Event has no duration")

        events = self._get_cache(self._events, self._event_key())

        if events is None:
            events = [LFPEvent(i, event_start, self) 
                      for i, event_start in enumerate(self.event_starts)]
                
            self._set_cache(self._events, self._event_key(), events,
                            self.max_event_cache)

        return events
    
    def get_valid_events(self):
        if not self.calc_opts.get('validate_events'):
            return self.events
        valid_events = self._get_cache(self._valid_events, self._event_key())
        if valid_events is None:
            valid_events = [e for e in self.events if e.is_valid]
            self._set_cache(self._valid_events, self._event_key(), valid_events,
                            self.max_event_cache)
        return valid_events
    
    def get_segments(self):
        segments = self._get_cache(self._segments, self._segment_key())
        if segments is None:
            # todo: it would be nice to warn or error here if you open a pkl with
            # a different structure than your current experiment
            ev_ok = self.get_event_validity(self.selected_brain_region)
            do_pad = self.calc_type not in ['coherence', 'psd']
            data = self.padded_data if do_pad else self.unpadded_data
            valid_sets = self.divide_data_into_valid_sets(data, ev_ok, do_pad=do_pad) 
            segments = [PeriodSegment(vs) for vs in valid_sets]
            self._set_cache(self._segments, self._segment_key(), segments,
                            self.max_segment_cache)
        return segments
                   
    def generate_spectrogram_cache_args(self, power_arg_set, calc_method):
        cache_args = deepcopy(power_arg_set)
        if isinstance(power_arg_set, dict):
            freqs = cache_args['freqs']
            n_cycles = cache_args['n_cycles']
            if is_iterable(freqs):
                cache_args['freqs'] = [freqs[0], freqs[1], freqs[1] - freqs[0]]
            if is_iterable(n_cycles):
                cache_args['n_cycles'] = [n_cycles[0], n_cycles[1], n_cycles[1] - n_cycles[0]]
            flat_args = [str(i) for item in list(cache_args.items()) for i in item]
        else:
            flat_args = [str(i) for i in cache_args]

        arg_set = [self.animal.identifier, self.selected_brain_region, calc_method, 
                   *flat_args, self.period_type, str(self.identifier), 'padding', 
                   *[str(pad) for pad in self.calc_opts.get('lfp_padding', [0, 0])]]
        return arg_set
    
    def calc_spectrogram(self):
        power_arg_set = self.calc_opts["power_arg_set"]
        calc_method = self.calc_opts.get("calc_method", "matlab")

        cache_args = self.generate_spectrogram_cache_args(power_arg_set, calc_method)
        saved_calc_exists, spectrogram, pickle_path = self.load(
            "lfp_output", "spectrogram", cache_args
        )
        if saved_calc_exists:
            return self.rebind_to_ureg(spectrogram)

        # --- run the calculation -------------
        if calc_method == "matlab":
            ml = MatlabInterface(self.env_config["matlab_config"])
            power, freqs, times = ml.mtcsg(self.padded_data, *power_arg_set)
            # `times` is already in seconds (numpy array)
        else:
            data_3d = self.padded_data[np.newaxis, np.newaxis, :]
            power = tfr_array_multitaper(data_3d, **power_arg_set).squeeze()
            freqs = power_arg_set["freqs"]
            dt = power_arg_set["decim"] / power_arg_set["sfreq"]  # seconds per bin
            times = np.arange(power.shape[-1]) * dt  # plain floats in seconds

        n_freq, n_time = power.shape

        # dim indices (unitless bins)
        freq_bin = np.arange(n_freq)
        time_bin = np.arange(n_time)

        # unitful coords
        spectrogram_time = self.standardize_time(
            self.quantity(
                times,
                units="second",
                dims=("time_bin",),
                name="spectrogram_time",
            )
        )

        frequency = self.quantity(
            freqs,
            units="Hz",
            dims=("freq_bin",),
            name="frequency",
        )

        da = xr.DataArray(
            power,
            dims=("freq_bin", "time_bin"),
            coords={
                "freq_bin": freq_bin,              # bin index (no units)
                "time_bin": time_bin,              # bin index (no units)
                "frequency": frequency,            # physical Hz, unitful
                "spectrogram_time": spectrogram_time,  # physical seconds, unitful
            },
            attrs={"calc_method": calc_method},
        )

        # --- time alignment / coords, all unitful ---
        true_beginning = (
            self.lfp_padding.sel(side="pre") - self.lost_signal.sel(side="pre")
        ).pint.to("second").reset_coords(drop=True)

        period_time = self.standardize_time(
            spectrogram_time - true_beginning,
            units="second",
        )

        absolute_time = self.standardize_time(
            spectrogram_time - true_beginning + self.onset,
            units="second",
        )

        da = da.assign_coords(
            period_time=period_time,
            absolute_time=absolute_time,
        )

        # --- infer bin size from spectrogram_time --
        if da.sizes["time_bin"] > 1:
            bin_size = (
                spectrogram_time.isel(time_bin=1)
                - spectrogram_time.isel(time_bin=0)
            ).pint.to("second")
        else:
            bin_size = None 

        da.attrs['bin_size'] = bin_size

        self.save(da, pickle_path)
        return da
    
    def get_power_(self):
        
        mask = (self.spectrogram.period_time >= 0) & (self.spectrogram.period_time < self.duration)
        power = self.spectrogram.sel(time_bin=mask)

        if self.events and self.calc_opts.get('validate_events'):
            ev_ok = self.get_event_validity(self.selected_brain_region)
            r_factor = self.to_int(self.event_duration/self.spectrogram.attrs['bin_size'])
            ev_mask = np.repeat(self.get_valid_vec(ev_ok, self), r_factor)
            power = power.where(ev_mask)
           
        if self.calc_opts.get("frequency_type") == "block":
            power = power.mean(dim="freq_bin", skipna=True)

        if self.calc_opts.get("time_type") == "block":
            power = power.mean(dim="time_bin", skipna=True)

        return power
        
    def index_transformation_function(self, concatenator):
        if concatenator == 'animal':
            return lambda calc: calc.assign_coords(
                time=calc.coords['time'] + self.onset
                ) if isinstance(calc, xr.DataArray) else calc
        else:
            raise NotImplementedError("Period concatenation is currently only supported by animal.")
        
    @property
    def segment_weights(self):
        if len(self.segments) == 0:
            return []
        return [len(seg.unpadded_data) for seg in self.segments]
    
    def get_psd_(self):
        return self.spectral_density_calc([self.unpadded_data], 'psd', welch_psd)


class PeriodSegment(Data, LFPMethods, SpectralDensityMethods):

    _name = 'period_segment'
    
    def __init__(self, data):
        self.unpadded_data = data

    def get_psd_(self):
        return self.spectral_density_calc(self.unpadded_data, 'psd', welch_psd)


class LFPEvent(Event, LFPMethods, LFPProperties):

    def __init__(self, identifier, onset, period):
        super().__init__(period, onset, identifier)
        self.animal = period.animal
        self.period_type = self.parent.period_type

    def __repr__(self):
        return (f"Event {self.animal.identifier} {self.period_type} "
                f"{self.period.identifier} {self.identifier}")

    @property          
    def is_valid(self):
        if f"Event {self.animal.identifier} {self.period_type} {self.period.identifier} {self.identifier}" == "Event INED18 pretone 4 0":
            a = 'foo'
        val = self.animal.lfp_event_validity[self.selected_brain_region][self.period_type][
            self.period.identifier][self.identifier]
        if not val:
            print(f"Event {self.animal.identifier} {self.period_type} {self.period.identifier} {self.identifier} is not valid!")
        return val
    
    @property
    def spectrogram(self):
        return self.parent.spectrogram
    
    @property
    def spectrogram_mask(self):
        eps   = self.quantity(1e-8, units='second')
        tbins = self.spectrogram.coords['period_time']
        rel_start = self.period.event_starts_in_period_time[self.identifier]
        win_start = rel_start - self.pre_event
        win_end   = rel_start + self.post_event
        mask = (tbins >= win_start - eps) & (tbins < win_end - eps)
        return mask

    def get_power_(self):

        # Boolean mask → subset times
        power = self.spectrogram.isel(time_bin=self.spectrogram_mask)

        # First time point in this window (period_time is already “time since period start”)
        event_start = power.coords["period_time"].isel(time_bin=0) + self.pre_event

        # Time relative to event onset
        power = power.assign_coords(
            relative_time=power.coords["period_time"] - event_start
        )

        if self.calc_opts.get("frequency_type") == "block":
            power = power.mean(dim="freq_bin")

        if self.calc_opts.get("time_type") == "block":
            power = power.mean(dim="time_bin")

        return power
    
    def index_transformation_function(self, concatenator):
        if concatenator == 'period':
            return lambda identifier, calc: calc.assign_coords(
                time=calc.coords['time'] + identifier * self.duration
                )
        elif concatenator == 'animal':
            return lambda identifier, calc: calc.assign_coords(
                time=calc.coords['time'] + identifier * self.duration + self.period.onset
                )
        else:
            raise NotImplementedError("Event concatenation is currently only supported by period and animal.")
    

class RegionRelationshipCalculator(Data, EventValidation, LFPProperties, DescendantCache):

    def __init__(self, period):
        self.period = period
        self.parent = self.period.parent
        self.period_id = period.identifier
        self.period_type = period.period_type
        self.identifier = f"{'_'.join(self.regions)}_{self.period.identifier}"
        self._segments = OrderedDict()
        self._events = OrderedDict()
        self.max_event_cache   = self.calc_opts.get('max_event_cache', 4)
        self.max_segment_cache = self.calc_opts.get('max_segment_cache', 4)

    @property
    def regions_data(self):
        processed_lfp = self.period.animal.processed_lfp
        start = self.to_int(self.period.start, unit='lfp_sample')
        stop = self.to_int(self.period.stop, unit='lfp_sample')
        return [processed_lfp[r][start:stop] for r in self.regions]

    @property
    def padded_regions_data(self):
        processed_lfp = self.period.animal.processed_lfp
        pad_start = self.to_int(self.period.pad_start, unit='lfp_sample')
        pad_stop = self.to_int(self.period.pad_stop, unit='lfp_sample')
        return [
            processed_lfp[r][pad_start:pad_stop]
            for r in self.regions
        ]
    
    @property
    def regions(self):
        return tuple(self.selected_brain_regions)

    @property
    def children(self):
        # TODO: 
        # it would be nice to have some logic in here to throw 
        # if children is events but events are not long enough for a valid 
        # calculation
        base = self.calc_opts.get('base', 'coherence_calculator')
        validate_events = self.calc_opts.get('validate_events')
        if validate_events:
            if 'event' in base:
                if self.pre_event + self.post_event == 0:
                    raise ValueError("Event has no duration")
                events = self.get_events()
                # Todo: this assumes that an event is long enough for multitaper.  This might not always be true.

                return [e for e in events if e.is_valid()]
            else:
                return self.get_segments()
        else:
            if 'calculator' in base:
                return []
            else:
                return self.get_events()

    def joint_event_validity(self):
        evs = [self.get_event_validity(region) for region in self.regions]
        return {i: all([ev[i] for ev in evs]) for i in evs[0]}
    
    def get_segments(self):

        segments = self._get_cache(self._segments, self._segment_key())

        if segments is None:

            ev_ok = self.joint_event_validity()
            do_pad = self.calc_type not in ['coherence', 'psd']
            regions_data = self.padded_regions_data if do_pad else self.regions_data

            valid_sets = list(zip(*(self.divide_data_into_valid_sets(data, ev_ok, do_pad=do_pad) 
                                for data in regions_data)))
            
            segments = [self.segment_class(self, vs, i) for i, vs in enumerate(valid_sets)]

            self._set_cache(self._segments, self._segment_key(), segments,
                                 self.max_segment_cache)
            
        return segments
        
    def get_events(self):
        event_key = self._event_key()
        events = self._get_cache(self._events, event_key)
        if events is None:
            events = [
                self.event_class(self.period, event_start, self, i)  # todo is this right
                for i, event_start in enumerate(self.period.event_starts)]
        return events
            
    def index_transformation_function(self, concatenator):
        raise NotImplementedError("Not yet implemented for RegionRelationshipCalculator.")
    
    @property
    def segment_weights(self):
        if len(self.segments) == 0:
            return []
        return [len(seg.regions_data[0]) for seg in self.segments]


class RelationshipCalculatorSegment(Data, LFPMethods, LFPProperties):

    def __init__(self, region_relationship_calculator, valid_data, index):
        self.region_relationship_calculator = region_relationship_calculator
        self.parent = region_relationship_calculator
        self.regions_data = valid_data
        self.regions = self.parent.regions
        self.period = self.parent.period
        self.identifier = index


class RelationshipCalculatorEvent(Event, LFPMethods, LFPProperties):

    def __init__(self, period, onset, region_relationship_calculator, index):
        super().__init__(period=period, onset=onset, index=index)
        self.region_relationship_calculator = region_relationship_calculator
        self.parent = region_relationship_calculator
        self.regions = self.parent.regions
        self.period = self.parent.period

    @property
    def regions_data(self):
        processed_lfp = self.period.animal.processed_lfp
        start = self.to_int((self.start - self.pre_event), unit='lfp_sample')
        stop = self.to_int((self.start + self.post_event), unit='lfp_sample')
        return [processed_lfp[r][start:stop] for r in self.regions]

    @property
    def padded_regions_data(self):
        processed_lfp = self.period.animal.processed_lfp
        start = self.to_int(
            self.start - self.pre_event - self.lfp_padding.sel(side="pre"), 
            unit='lfp_sample')
        stop = self.to_int(
            self.start + self.pre_event + self.lfp_padding.sel(side="post"), 
            unit='lfp_sample')
        return [processed_lfp[r][start:stop] for r in self.regions]
    

class CoherenceCalculatorSegment(RelationshipCalculatorSegment, 
                                 CoherenceMethods, LFPMethods):

    _name = 'coherence_segment'

    def __init__(self, coherence_calculator, data, index):
        super().__init__(coherence_calculator, data, index)


class CoherenceCalculatorEvent(RelationshipCalculatorEvent, CoherenceMethods, 
                               LFPMethods):
    _name = 'coherence_event'

    def __init__(self, period, onset, region_relationship_calculator, index):
        super().__init__(period, onset, region_relationship_calculator, index)


class CoherenceCalculator(RegionRelationshipCalculator, CoherenceMethods, LFPMethods):

    _name = 'coherence_calculator'
    segment_class = CoherenceCalculatorSegment
    event_class = CoherenceCalculatorEvent
    
   
class AmpXCorrCalculator(RegionRelationshipCalculator, BandPassFilterMixin):

    _name = 'amp_xcorr_calculator'

    def __init__(self, period):
        super().__init__(period)

    @property
    def lags(self):
        if self.calc_type == 'lag_of_max_corr':
            raise ValueError("Data type is max correlation; there are not multiple lags.")
        return [TimeBin(i, data_point, self) for i, data_point in enumerate(self.calc)]
    
    def get_lag_of_max_corr(self):
        # TODO: it is causing all kinds of special pleading elsewhere in the code to have this method
        # I think we should move toward letting the user have a list of operations on the base calc type
        # they can specify, argmax and argmin being two of them.  I mean, arguably that's what `histogram`
        # already is, but it's also an `aggregate`.  I mean just mathematical operations
        amp_xcorr = self.get_amp_xcorr()
        if contains_nan(amp_xcorr):
            return xr.DataArray(np.nan)
        return amp_xcorr.idxmax('lag').item()

    def amplitude(self, signal):
        pad_len = self.to_int(self.lfp_padding)
        env = np.abs(apply_hilbert_to_padded_data(self.filter(signal), pad_len))
        # gentle taper; alpha=0.2 is mild, won’t distort the middle
        return env * tukey(env.size, alpha=0.2)
     
    def get_amp_xcorr(self):
        fs = self.to_float(self.lfp_sampling_rate, unit="Hz")

        if not self.calc_opts.get('validate_events'):
            n1, n2 = (len(self.padded_regions_data[0]), len(self.padded_regions_data[1]))
            assert n1 == n2, f"LFP regions mismatch: {n1} vs {n2}"

            amp1, amp2 = [self.amplitude(signal) for signal in self.padded_regions_data]
            corr, _ = normalized_xcorr(amp1, amp2, fs=fs)
            result = corr

        else:
            ev_ok = self.get_event_validity()
            valid_sets, len_sets = self.get_valid_sets(ev_ok, do_pad=True)
            if not valid_sets:
                return xr.DataArray(np.nan)

            len_longest_corr = max(len_sets) * 2 - 1
            corrs = []

            for signal_series in valid_sets:
                amp1, amp2 = [self.amplitude(signal) for signal in signal_series]
                corr, _ = normalized_xcorr(amp1, amp2, fs=fs)
                # todo: this len_longest_corr logic will cause an error if you
                # are trying to average over different calculators with different lengths
                # in practice, often not a problem since you restrict the length of the longest corr
                # but should be fixed
                weight = corr.size / len_longest_corr
                padded = np.full(len_longest_corr, np.nan)
                start = (len_longest_corr - corr.size) // 2
                padded[start:start + corr.size] = corr * weight
                corrs.append(padded)

            result = np.nansum(np.vstack(corrs), axis=0)

        # Build lags deterministically from the final result length
        assert result.size % 2 == 1, "Cross-corr length should be odd (2*N-1)."
        half = (result.size - 1) // 2
        lags = np.arange(-half, half + 1, dtype=float) / fs

        max_lag_sec = self.calc_opts.get('max_lag_sec', None)
        if max_lag_sec is None and 'lags' in self.calc_opts:
            max_lag_sec = self.calc_opts['lags'] / fs
        if max_lag_sec is not None:
            m = (lags >= -max_lag_sec) & (lags <= max_lag_sec)
            result = result[m]
            lags = lags[m]

        return xr.DataArray(result, dims=['lag'], coords={'lag': lags})
    
    
class GrangerCalculator(RegionRelationshipCalculator): 

    name = 'granger_calculator'


    def get_granger_model_order(self):
        return self.granger_stat('ts_data_to_info_crit')
    
    def get_granger_diagnostics(self):
        ml = MatlabInterface(self.env_config['matlab_config'])
        fres_req = self.calc_opts.get('fres_req', 500)
        result = ml.mvgc_run_with_diag(np.vstack(self.regions_data), fres_req, momax=100)
        diag = json.loads(result[0]['result'][0].decode('utf-8'))
        return diag
    
    def get_granger_causality(self):
        ids = [self.period.animal.identifier, self.period_type, str(self.period.identifier), 
               str(self.selected_frequency_band)]
        saved_calc_exists, saved_granger_calc, pickle_path = self.load('lfp_output', 'granger', ids)
        if saved_calc_exists:
            return saved_granger_calc
        ml = MatlabInterface(self.env_config['matlab_config'])
        result = ml.granger_causality(self.regions_data)

        a = 'foo'


    
