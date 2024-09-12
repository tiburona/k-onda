class SpikeMethods:
    
    def select_spike_children(self):
        if self.selected_neuron_type:
            return getattr(self, self.selected_neuron_type)
        else: 
            return self.units['good']

    def get_psth(self):
        return self.get_average('get_psth', stop_at=self.calc_opts.get('base', 'event'))
    
    def get_firing_rates(self):
        return self.get_average('get_firing_rates', stop_at=self.calc_opts.get('base', 'event'))