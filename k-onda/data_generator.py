import os
import pandas as pd
import json
from copy import deepcopy
import numpy as np
from base_data import Base

class DataGenerator(Base):

    base_levels = {
        'psth': 'event',
        'firing_rates': 'event'
    }

    def __init__(self, experiment):
        self.experiment = experiment
        self.opts_dicts = []
        

    def check_for_preexisting_file(self):
        file_path = os.path.join(self.file_path, self.data_type, self.experiment_id)
        csv_path = file_path + '.csv'
        json_path = file_path + '.json'
        if os.path.exists(csv_path) and os.path.exists(json_path):
            with open(json_path, 'r') as json_file:
                file_opts = json.load(json_file)
                essential_opts = self.get_essential_data_opts()
                for key in essential_opts:
                    if key in file_opts and file_opts[key] == essential_opts[key]:
                        continue
                    else:
                        return None

            return pd.read_csv(csv_path)
        else:
            return None

    def get_essential_data_opts(self):
        # this should iterate through data opts and return key val pairs for which, if they don't 
        # match, data will have to be regenerated.
        pass


    # TODO: this needs to be called first, in the prep step to running
    def set_attributes(self, data_opts):
        self.data_opts = data_opts
        if self.data_class == 'lfp':
            fb = self.current_frequency_band
            if not isinstance(self.current_frequency_band, str):
                translation_table = str.maketrans({k: '_' for k in '[](),'})
                fb = str(list(fb)).translate(translation_table)

    def generate_data(self):
        self.opts_dicts.append(deepcopy(self.data_opts))

        key = self.set_data_key()  # TODO Here I should add the opts
        level = self.base_levels[self.data_type]
        sources = getattr(self.experiment, f'all_{level}s')

    def initialize_data_dicts(self):
        level = self.base_levels[self.data_type]
        sources = getattr(self.experiment, f'all_{level}s')
        




    def make_df(self):
        self.opts_dicts.append(deepcopy(self.data_opts))
        name = self.set_df_name()
        df = pd.DataFrame(self.get_rows())
        vs = ['unit_num', 'animal', 'category', 'group', 'frequency']
        for var in vs:
            if var in df:
                df[var] = df[var].astype('category')
        if name in self.experiment.data_frames:
            name += '_2'
        self.experiment.data_frames[name] = df

    def set_data_key(self):
        name = self.data_type
        if 'lfp' in self.data_class:
            name += f"_{self.current_brain_region}_{self.current_frequency_band}"
        return name

    def get_rows(self):
        """
        Prepares the necessary parameters and then calls `self.get_data` to collect rows of data based on the specified
        level, attributes, and inclusion criteria.

        The function determines the level (i.e., the object type) from which data will be collected, the additional
        attributes to be included in each row, and the criteria that an object must meet to be included in the data.

        Parameters:
        None

        Returns:
        list of dict: A list of dictionaries, where each dictionary represents a row of data. The keys in the dictionary
        include the data column, source identifiers, ancestor identifiers, and any other specified attributes.

        """
        level = self.base_levels[self.data_type]
        
        other_attributes = ['period_type']
        
        if 'lfp' in self.data_class:
            if self.data_type in ['mrl']:
                other_attributes += ['frequency', 'fb', 'neuron_type', 'neuron_quality']  # TODO: figure out what fb should be changed to post refactor
            if level == 'granger_segment':
                other_attributes.append('length')
            if any([w in self.data_type for w in ['coherence', 'correlation', 'phase', 'granger']]):
                other_attributes.append('period_id')
        else:
            other_attributes += ['category', 'neuron_type', 'quality']

        return self.get_data(level, other_attributes)
    

    def get_data(self, level, other_attributes):
        """
        Collects data from specified data sources based on the provided level and criteria. The function returns a list
        of dictionaries, where each dictionary represents a row of data. Each row dictionary contains data values,
        identifiers of the source, identifiers of all its ancestors, and any other specified attributes of the source
        or its ancestors.

        Parameters:
        - level (str): Specifies the hierarchical level from which data should be collected. This determines which
          sources are considered for data collection.
        - inclusion_criteria (list of callables): A list of functions that each take a source as an argument and return
          a boolean value. Only sources for which all criteria functions return True are included in the data
          collection.
        - other_attributes (list of str): A list of additional attribute names to be collected from the source or its
          ancestors. If an attribute does not exist for a particular source, it will be omitted from the row dictionary.

        Returns:
        list of dict: A list of dictionaries, where each dictionary represents a row of data. The keys in the dictionary
        include the data column, source identifiers, ancestor identifiers, and any other specified attributes.

        Notes:
        - The function first determines the relevant data sources based on the specified `level` and the object's
          `data_class` attribute.
        - If the `frequency_type` in `data_opts` is set to 'continuous', the function further breaks down the sources
          based on frequency bins.
        - Similarly, if the `time_type` in `data_opts` is set to 'continuous', the sources are further broken down based
          on time bins.
        - The final list of sources is filtered based on the provided `inclusion_criteria`.
        - For each source, a row dictionary is constructed containing the data, source identifiers, ancestor
          identifiers, and any other specified attributes.
        """

        rows = []
        if self.data_class == 'lfp':
            experiment = self.lfp
        elif self.data_class == 'behavior':
            experiment = self.behavior
        else:
            experiment = self.experiment

        if level in ['event', 'period']:
            level = self.data_class + '_' + level

        if self.data_class == 'spike':
            for unit in self.experiment.all_units:
                unit.prepare_periods()
        elif self.data_class == 'lfp':
            for animal in self.experiment.all_animals:
                animal.prepare_periods()


        # TODO: one way for this to work for both base calculations and making csv files that only have
        # evoked etc is to be able to either invoke calculate_data() or the data property

        sources = getattr(experiment, f'all_{level}s')

        calcs = [(source, getattr(source, f"get_{self.data_type}")()) for source in sources]



        if self.data_opts.get('frequency_type') == 'continuous':
            other_attributes.append('frequency')
            calcs = [(frequency_bin, frequency_bin.data) for source, data in calcs 
                       for frequency_bin in source.get_frequency_bins(data)]
        if self.data_opts.get('time_type') == 'continuous':
            other_attributes.append('time')
            calcs = [(time_bin, time_bin.data) for source, data in calcs 
                       for time_bin in source.get_time_bins(data)]
        
        if self.data_class == 'lfp':
            if any([s in self.data_type for s in ['coherence', 'correlation', 'phase', 'granger']]):
                self.data_col = f"{
                    self.data_opts['region_set']}_{self.current_frequency_band}_{self.data_type}"
            else:
                self.data_col = f"{
                    self.current_brain_region}_{self.current_frequency_band}_{self.data_type}"
        else:
            self.data_col = self.data_type
      
        for source, data in calcs:
            if self.data_opts.get('aggregator') == 'sum':
                row_dict = {self.data_col: np.sum(data)}
            elif self.data_opts.get('aggregator') == 'none':
                if isinstance(data, dict):
                    row_dict = {f"{self.data_col}_{key}": val for key, val in source.data.items()}
                else:
                    row_dict = {self.data_col: data}
            else:
                row_dict = {self.data_col: np.mean(data)}
            
            for src in source.ancestors:
                row_dict[src.name] = src.identifier
                for attr in other_attributes:
                    val = getattr(src, attr) if hasattr(src, attr) else None
                    if val is not None:
                        if attr == 'period_id':
                            attr = 'period'
                        row_dict[attr] = val
            rows.append(row_dict)

        return rows