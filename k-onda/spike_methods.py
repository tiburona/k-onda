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
    
    def construct_combination_unit_type(self):
        # a placeholder for something you can imagine someone wanting to implement
        # it would need to take configuration specs from experiment info, then assign a property to 
        # animal that was the concatenation of two or more unit lists
        pass