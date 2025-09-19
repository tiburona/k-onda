class LFPMethods:
 
    def get_power(self, exclude=True):
        return self.get_average('get_power', stop_at=self.calc_opts.get('base', 'event'), 
                                exclude=exclude)
    
    def get_coherence(self):
        return self.get_average('get_coherence', 
                                stop_at=self.calc_opts.get('base', 'coherence_calculator'))
    
    def get_amp_crosscorr(self):
        return self.get_average('get_amp_xcorr', 
                                stop_at=self.calc_opts.get('base', 'amp_xcorr_calculator'))
    
    def get_lag_of_max_corr(self):
        return self.get_average('get_lag_of_max_corr', 
                                stop_at=self.calc_opts.get('base', 'amp_xcorr_calculator'))