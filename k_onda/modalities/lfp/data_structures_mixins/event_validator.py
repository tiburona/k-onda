import numpy as np
from k_onda.utils import to_hashable


class DescendentCache:

    def _get_cache(self, cache, key):
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        return None

    def _set_cache(self, cache, key, value, max_size):
        cache[key] = value
        cache.move_to_end(key)
        if len(cache) > max_size:
            cache.popitem(last=False)  # drop oldest

    def _base_key(self):
        region_key = self.selected_brain_region if self.name == 'period' else self.selected_brain_regions
        band   = self.selected_frequency_band if self.selected_frequency_band is not None else None
        return (region_key, band)

    def _segment_key(self):
        return to_hashable(self._base_key())
       
    def _event_key(self):

        ev_cfg = self.calc_opts.get('events', {}).get(self.period_type, {})
        pre  = ev_cfg.get('pre_event', self.pre_event)
        post = ev_cfg.get('post_event', self.post_event)

        return to_hashable(self._base_key() + (pre, post))
    
    def _cached_collection(self, cache_name, key_fn, build_fn, max_size):
        """
        Generic helper for events/segments-style cached collections.
        cache_name: str like '_events' or '_segments'
        key_fn:     callable(self) -> hashable key
        build_fn:   callable(self) -> value to cache
        max_size:   int, max entries for this cache
        """
        cache = getattr(self, cache_name)
        key = key_fn()
        cached = self._get_cache(cache, key)
        if cached is not None:
            return cached

        value = build_fn()
        self._set_cache(cache, key, value, max_size)
        return value
    
    @property
    def events(self):
        return self._cached_collection(
            cache_name="_events",
            key_fn=self._event_key,
            build_fn=self.get_events,
            max_size=self.max_event_cache,
        )

    @property
    def segments(self):
        return self._cached_collection(
            cache_name="_segments",
            key_fn=self._segment_key,
            build_fn=self.get_segments,   
            max_size=self.max_segment_cache,
        )

class EventValidation:
    
    def get_event_validity(self, region):
        period = self if self.name == 'period' else self.period
        ev = period.animal.lfp_event_validity[region]
        return {i: valid for i, valid in enumerate(ev[self.period_type][period.identifier])}
    
    def get_valid_vec(self, ev_ok, period):
        valid_vec = np.zeros(n_events, dtype=bool)
        n_events = len(period.event_starts_in_period_time)
        valid_vec = np.zeros(n_events, dtype=bool)
        for k in range(n_events):
            valid_vec[k] = ev_ok.get(k, False)
    
        return valid_vec
    
    def divide_data_into_valid_sets(self, data, ev_ok, do_pad=False):
       
        if not self.calc_opts.get('validate_events'):
            return [np.asarray(data)] 
        
        period = self if self.name == 'period' else self.period
        
        n_events = len(period.event_starts_in_period_time)

        valid_vec = self.get_valid_vec(ev_ok, period)

        # Find contiguous runs of True
        runs = []
        i = 0
        while i < n_events:
            if not valid_vec[i]:
                i += 1
                continue
            j = i
            while j < n_events and valid_vec[j]:
                j += 1
            # events [i, j) are valid
            runs.append((i, j))
            i = j

        valid_sets = []
        # pad lengths in samples
        padL, padR = int(self.lfp_padding[0]), int(self.lfp_padding[1])

        if self.calc_opts.get('min_len'):
            min_len = self.calc_opts['min_len'] * self.lfp_sampling_rate
        else:
            if self.calc_type == 'coherence':
                coherence_args = self.welch_and_coherence_args('coherence')
                nperseg = coherence_args['nperseg']
                min_len = nperseg
            else:
                min_len = (self.lfp_sampling_rate + 1) * 3 + 1

        event_start_times = self.to_float(
            period.event_starts_in_period_time, 
            units='lfp_sample')
        
        event_duration = self.event_duration.pint.to('lfp_sample')

        for ev_start, ev_stop in runs:
            # these times need to be relative to the period start
            start_samp = int(event_start_times[ev_start])  
            stop_samp  = int(event_start_times[ev_stop - 1]) + event_duration
            if do_pad:
                left  = max(0, start_samp - padL)
                right = min(len(data), stop_samp + padR)
            else:
                left, right = start_samp, stop_samp

            seg = np.asarray(data[left:right])
            if seg.size > min_len:
                valid_sets.append(seg)

        return valid_sets