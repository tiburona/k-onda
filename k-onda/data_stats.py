import numpy as np

class Data:

    summarizers = {'psth': np.mean, 'proportion': np.mean, 'spike_count': np.sum}
    transformations = {'psth': {}}
    
    current_filters = {}


    @property
    def data(self):
        data = self.get_data()
        if self.data_opts.get('evoked'): # have to figure out a way to do evoked that isn't circular
            data -= self.get_reference_data()
        
        
        # TODO: Maybe I need a concept transformation in here to deal with the fact that for psth
        # I need to do one last thing, which is divide by std-dev.
        
    def get_reference_data(self, data_type=None):
        if data_type is None:
            data_type = self.data_type
        current_period_type = self.current_filters['period_type']
        self.current_filters['period_type'] = self.experiment.info['reference'][current_period_type]
        reference_data = self.get_data(data_type)
        self.current_filters['period_type'] = current_period_type
        return reference_data


    def get_data(self, data_type=None):
        
        if data_type is None:
            data_type = self.data_type

        data_frame = self.experiment.data_frames.get(data_type)
        
        if data_frame is None:
            self.data_generator.make_df()
            data_frame = self.experiment.data_frames.get(self.data_type)

        data = self.summarize_from_data_frame(data_frame)
        return data
    
    def calculate_data(self):
        return getattr(self, f"get_{self.data_type}")()
    
    def summarize_from_data_frame(self, summarizer):
        summarizer = self.summarizers[self.data_type]
        data = self.experiment.data_frames[self.data_type]
        data = self.filter_data_frame(data)
        for level in reversed(self.hierarchy):
            # group by all levels above this one
            if level == self.name:
                break
            group_levels = self.hierarchy[0:self.hierarchy.find(level)]
            grouped_data = data.groupby(group_levels)
            data = grouped_data.apply(summarizer).reset_index()
        return data
    
    def filter_data_frame(self, data):
        # common filters: neuron_type, period_type, frequency, neuron quality
        # expected form of filter
        # {'period_type': 'tone'}

        for key, val in self.current_filters.items():
            data = data[data[key] == val]  
            return data

        
 

