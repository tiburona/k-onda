class LFPDataSelector:
    """A class with methods shared by LFPPeriod and LFPEvent that are used to return portions of their data."""

    @property
    def lost_signal(self):
        return self.calc_opts.get('lost_signal', [0, 0])

    def slice_spectrogram(self):  

        # TODO make this use the special method I wrote for frequency selection  
                                    
        tol = 0.2                           

        # boolean mask along the *frequency* coord
        fmask = ((self.spectrogram['frequency'] >= self.freq_range[0] - tol) &
                (self.spectrogram['frequency'] <= self.freq_range[1] + tol))         

        return self.spectrogram.sel(frequency=fmask)


    @property
    def sliced_spectrogram(self):
        return self.slice_spectrogram()