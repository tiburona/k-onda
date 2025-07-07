import pandas as pd
import csv
import os
from copy import deepcopy
import json
import random
import string
from functools import reduce
from xarray import DataArray

from k_onda.core import OutputGenerator
from k_onda.utils import safe_make_dir, to_serializable


class CSVTabulator(OutputGenerator):
    """A class to construct dataframes and write out csv files."""
    def __init__(self, experiment):
        super().__init__()
        self.experiment = experiment
        self.dfs = []
        self.data_col = None
        self.spreadsheet_fname = None
        self.results_path = None
        self.script_path = None
        self.opts_dicts = []
        self.name_suffix = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(6))

    def initialize(self):
        """Both initializes values on self and sets values for the context."""
        self.experiment.initialize_data() 
        self.set_attributes()

    def set_attributes(self):
        if self.kind_of_data in ['lfp', 'mrl']:
            fb = self.selected_frequency_band
            if not isinstance(self.selected_frequency_band, str):
                translation_table = str.maketrans({k: '_' for k in '[](),'})
                fb = str(list(fb)).translate(translation_table)
            if any([s in self.calc_type for s in ['coherence', 'correlation', 'phase', 'granger']]):
                self.data_col = f"{self.calc_opts['region_set']}_{fb}_{self.calc_type}"
            else:
                self.data_col = f"{self.selected_brain_region}_{fb}_{self.calc_type}"
        else:
            self.data_col = 'rate' if self.calc_type == 'psth' else self.calc_type

    def make_df(self):
        self.initialize()
        self.opts_dicts.append(deepcopy(self.calc_opts))
        df = pd.DataFrame(self.get_rows())
        vs = ['unit_num', 'animal', 'category', 'group', 'frequency']
        for var in vs:
            if var in df:
                df[var] = df[var].astype('category')
        self.dfs.append(df)

    def merge_dfs_animal_by_animal(self):

        # Extract unique animals across all data frames
        unique_animals = pd.concat([df['animal'] for df in self.dfs if 'animal' in df.columns]).unique()

        # List to hold merged data for each animal
        merged_data_per_animal = []

        for animal in unique_animals:
            # Filter DataFrames for the current animal, remove 'animal' column
            dfs_per_animal = [df[df['animal'] == animal].copy().drop(columns=['animal']) 
                              for df in self.dfs if animal in df['animal'].values]

            # Adjust DataFrames to prioritize 'time' over 'time_bin'
            for df in dfs_per_animal:
                if 'time' in df.columns and 'time_bin' in df.columns:
                    df.drop(columns='time_bin', inplace=True)

            if not dfs_per_animal:
                continue

            # Use reduce to merge data frames progressively, ensuring only common columns are used at each step
            def merge_dfs(left, right):
                common_columns = left.columns.intersection(right.columns)
                return pd.merge(left, right, on=list(common_columns), how='outer')

            merged_animal_df = reduce(merge_dfs, dfs_per_animal)

            # Add the animal identifier back to the merged data
            merged_animal_df['animal'] = animal

            # Append to the list
            merged_data_per_animal.append(merged_animal_df)

        # Concatenate all merged data
        final_merged_df = pd.concat(merged_data_per_animal, ignore_index=True)

        # Store or return the final merged DataFrame
        self.dfs.append(final_merged_df)
        return final_merged_df

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
        level = self.calc_opts['row_type']
        
        other_attributes = ['period_type']
        
        if 'lfp' in self.kind_of_data:
            if level == 'granger_segment':
                other_attributes.append('length')
            if any([w in self.calc_type for w in ['coherence', 'correlation', 'phase', 'granger']]):
                other_attributes.append('period_id')
            else:
                if self.calc_opts['time_type'] == 'continuous' and self.calc_opts.get('power_deviation'):
                    other_attributes.append('power_deviation')
        elif 'mrl' in self.kind_of_data:
            level = 'mrl_calculator'
            other_attributes += ['frequency', 'neuron_type', 'neuron_quality']  
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
          `kind_of_data` attribute.
        - If the `frequency_type` in `calc_opts` is set to 'continuous', the function further breaks down the sources
          based on frequency bins.
        - Similarly, if the `time_type` in `calc_opts` is set to 'continuous', the sources are further broken down based
          on time bins.
        - The final list of sources is filtered based on the provided `inclusion_criteria`.
        - For each source, a row dictionary is constructed containing the data, source identifiers, ancestor
          identifiers, and any other specified attributes.
        """

        rows = []

        sources = [source for source in getattr(self.experiment, f'all_{level}s') if source.include()]

        if self.calc_opts.get('frequency_type') == 'continuous':
            other_attributes.append('frequency')
            sources = [frequency_bin for source in sources for frequency_bin in source.frequency_bins]
        if self.calc_opts.get('time_type') == 'continuous':
            other_attributes.append('time')
            sources = [time_bin for source in sources for time_bin in source.time_bins]

        attr = self.calc_opts.get('attr', 'mean')

        for source in sources:

            result = float(getattr(source, attr).values)
            if isinstance(result, dict):
                row_dict = {f"{self.data_col}_{key}": val for key, val in result.items()}
            else:
                row_dict = {self.data_col: result}
            
            for src in source.ancestors:
                row_dict[src.name] = src.identifier

            for other_attr in other_attributes:
                found_attr = False
                for src in source.ancestors:
                    val = getattr(src, other_attr, None)
                    if val is not None:
                        if isinstance(val, DataArray):
                            val = val.values
                        row_dict[other_attr] = val
                        found_attr = True
                        break
                if found_attr:
                    continue

            rows.append(row_dict)

        return rows

    def make_csv(self, opts):
        
        expanded_calc_opts = opts['calc_opts']

        for opts in expanded_calc_opts:
            self.calc_opts = opts
            self.make_df()

        self.merge_dfs_animal_by_animal()
        self.write_csv()

    def write_csv(self):

        self.build_write_path('csv')

        force_recalc = self.io_opts.get('force_recalc', True)

        if os.path.exists(self.file_path) and not force_recalc:
            return
        
        safe_make_dir(self.file_path)

        with open(self.file_path, 'w', newline='') as f:

            for opts_dict in self.opts_dicts:
                safe_dict = {k: to_serializable(v) for k, v in opts_dict.items()}
                f.write("# " + json.dumps(safe_dict, separators=(", ", ": ")) + "\n")

            df = self.dfs[-1]
            header = list(df.columns)
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for index, row in df.iterrows():
                writer.writerow(row.to_dict())


