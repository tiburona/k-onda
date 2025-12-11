from k_onda.utils import to_hashable, safe_get


class DescendantCache:

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
        region_key = self.selected_brain_region or self.selected_brain_regions
        band   = self.selected_frequency_band 
        return (region_key, band)

    def _segment_key(self):
        return to_hashable(self._base_key())
       
    def _event_key(self):
        period = self.period if getattr(self, 'period', None) else self
        ev_cfg = safe_get(
            self.calc_opts, [period.period_type, 'event_pre_post'], default=())
        return to_hashable(self._base_key() + ev_cfg)
    
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