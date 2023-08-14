import subprocess
import pandas as pd
import csv
import os

from data import Base
from lfp import LFPAnimal, FrequencyPeriod, FrequencyBin, TimeBin as FrequencyTimeBin, FrequencyUnit, initialize_lfp
from spike import Group, Animal, Unit, Period, Trial, TimeBin

LEVEL_DICT = dict(animal=Animal, group=Group, unit=Unit, period=Period, trial=Trial, frequency_period=FrequencyPeriod)


class Stats(Base):
    """A class to construct dataframes, write out csv files, and call R for statistical tests."""
    def __init__(self, experiment, data_type_context, neuron_type_context, data_opts):
        self.experiment = experiment
        self.data_type_context = data_type_context
        self.data_opts = data_opts
        self.neuron_type_context = neuron_type_context
        self.selected_neuron_type = None
        self.dfs = {}
        self.time_type = self.data_opts.get('time')
        self.data_col = None
        self.num_time_points = None
        self.spreadsheet_fname = None
        self.results_path = None
        self.script_path = None
        self.lfp_brain_region = None
        self.frequency_bands = None
        self.current_frequency_band = None

    @property
    def rows(self):
        return self.get_rows()

    def set_attributes(self):
        self.data_col = 'rate' if self.data_type == 'psth' else self.data_type
        if self.data_class == 'lfp':
            self.data_col = f"{self.current_frequency_band}_{self.data_type}"
            self.lfp_brain_region = self.data_opts['brain_region']
        else:
            self.data_col = 'rate' if self.data_type == 'psth' else self.data_type
            pre_stim, post_stim, bin_size = (self.data_opts[k] for k in ('pre_stim', 'post_stim', 'bin_size'))
            self.num_time_points = int((pre_stim + post_stim) / bin_size)

    def smart_merge(self, keys):
        # Determine which DataFrame has the most unique key combinations
        df_items = list(self.dfs.items())  # get a list of (name, DataFrame) pairs
        df_items.sort(key=lambda item: item[1][list(keys)].drop_duplicates().shape[0], reverse=True)

        # Extract the sorted DataFrames and their names
        df_names, dfs = zip(*df_items)

        # Start with the DataFrame that has the most unique key combinations
        result = dfs[0]

        # Merge with all the other DataFrames
        for df in dfs[1:]:
            if 'experiment' in df.columns:
                df = df.drop(columns=['experiment'])
            result = pd.merge(result, df, how='left', on=keys)

        new_df_name = '_'.join(df_names)
        self.dfs[new_df_name] = result
        return new_df_name

    def make_dfs(self, opts_dicts):
        name = self.data_type
        for opts in opts_dicts:
            self.data_opts = opts
            if 'fb' in self.data_opts:
                self.frequency_bands = self.data_opts['fb']
                for fb in self.data_opts['fb']:
                    self.current_frequency_band = fb
                    self.make_df(f"{name}_{fb}")
            else:
                self.make_df(name)
        common_columns = set(list(self.dfs.values())[0].columns)
        for df in self.dfs.values():
            common_columns &= set(df.columns)
        common_columns = [col for col in common_columns if col != 'experiment']
        new_name = self.smart_merge(list(common_columns))
        return new_name

    def make_df(self, name):
        self.set_attributes()
        self.initialize_data()
        df = pd.DataFrame(self.rows)
        vs = ['unit_num', 'animal', 'category', 'condition', 'two_way_split', 'three_way_split', 'frequency']
        for var in vs:
            if var in df:
                df[var] = df[var].astype('category')
        self.dfs[name] = df

    def initialize_data(self):
        if 'lfp' in self.data_class:
            initialize_lfp()
            if self.data_opts['frequency'] == 'continuous' or self.data_type == 'mrl':  # TODO: change the name of row_type
                FrequencyPeriod.initialize_data()
        if self.data_opts['time'] == 'continuous':
            LEVEL_DICT[self.data_opts['row_type']].initialize_data()  # TimeBins  aren't created automatically

    def get_rows(self, inclusion_criteria=None, other_attributes=None):
        if other_attributes is None:
            other_attributes = []
        if inclusion_criteria is None:
            inclusion_criteria = []
        if 'lfp' in self.data_class:
            if self.data_type == 'mrl':
                level = FrequencyUnit
                other_attributes += ['frequency', 'fb', 'category']
            else:
                level = FrequencyBin if self.data_opts['frequency'] == 'continuous' else FrequencyPeriod
                inclusion_criteria.append(lambda x: x.fb == self.current_frequency_band)
        else:
            level = TimeBin if self.data_opts['time'] == 'continuous' else Period
            other_attributes += ['category']
            parent = LEVEL_DICT[self.data_opts['row_type']]
            inclusion_criteria.append(lambda x: isinstance(x.parent, parent))
        other_attributes += ['period_type']

        rows = self.get_data(level, inclusion_criteria, other_attributes)
        level.instances.clear()

        return rows

    def get_data(self, level, inclusion_criteria, other_attributes):
        rows = []
        instances = [instance for instance in level.instances
                     if all(criterion(instance) for criterion in inclusion_criteria)]
        for instance in instances:
            row_dict = {self.data_col: instance.data}
            current_instance = instance
            while True:
                row_dict[current_instance.name] = current_instance.identifier
                for attr in other_attributes:
                    val = getattr(current_instance, attr) if hasattr(current_instance, attr) else None
                    if val is not None:
                        row_dict[attr] = val  # can use this for stage
                if hasattr(current_instance, 'parent') and current_instance.parent is not None:
                    current_instance = current_instance.parent  # Go one level up to the parent
                else:
                    break  # Stop the loop if there is no parent
            rows.append(row_dict)
        return rows

    def make_spreadsheet(self, df_name=None):
        row_type = self.data_opts['row_type']
        path = self.data_opts.get('data_path')
        df_name = df_name if df_name else self.data_type
        name = df_name
        if df_name not in self.dfs:
            self.make_df(df_name)
        if 'lfp' in name:
            path = os.path.join(path, 'lfp')
            name += f"_{self.lfp_brain_region}"
        fname = os.path.join(path, '_'.join([name, self.time_type, row_type + 's']) + '.csv')
        self.spreadsheet_fname = fname

        with open(fname, 'w', newline='') as f:
            header = list(self.dfs[df_name].columns)
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()

            for index, row in self.dfs[df_name].iterrows():
                writer.writerow(row.to_dict())

    def write_r_script(self):

        error_suffix = '/unit_num' if self.data_opts['row_type'] == 'trial' else ''
        error_suffix = error_suffix + '/time_bin' if self.data_opts['post_hoc_bin_size'] > 1 else error_suffix

        if self.data_opts['post_hoc_type'] == 'beta':
            model_formula = f'glmmTMB(formula = {self.data_col} ~ condition +  (1|animal{error_suffix}), family = beta_family(link = "logit"), data = data)'
            interaction_model_formula = f'glmmTMB(formula = {self.data_col} ~ condition * category + (1|animal{error_suffix}), family = beta_family(link = "logit"), data = sub_df)'
            p_val_index = 'coefficients$cond[2, 4]'
            interaction_p_val_index = 'coefficients$cond["conditionstressed:categoryPN", 4]'
            zero_adjustment_line = f'df$"{self.data_col}"[df$"{self.data_col}" == 0] <- df$"{self.data_col}"[df$"{self.data_col}" == 0] + 1e-6'
        elif self.data_opts['post_hoc_type'] == 'lmer':
            model_formula = f'lmer({self.data_col} ~ condition +  (1|animal{error_suffix}), data = data)'
            interaction_model_formula = f'lmer({self.data_col} ~ condition * category + (1|animal{error_suffix}), data = sub_df)'
            p_val_index = 'coefficients[2, 5]'
            interaction_p_val_index = 'coefficients[4, 5]'
            zero_adjustment_line = ''

        r_script = fr'''
        library(glmmTMB)
        library(lme4)
        library(lmerTest)  # for lmer p-values
        library(readr)

        df <- read_csv('{self.spreadsheet_fname}')

        # Convert variables to factors
        factor_vars <- c('unit_num', 'animal', 'category', 'condition')
        df[factor_vars] <- lapply(df[factor_vars], factor)

        # Add small constant to 0s in the data column, if necessary
        {zero_adjustment_line}

        # Create an empty data frame to store results
        results <- data.frame()

        perform_regression <- function(data, category){{
            # Perform regression with mixed effects
            mixed_model <- {model_formula}

            # Extract p-value
            p_val <- summary(mixed_model)${p_val_index}
            return(p_val)
        }}

        # Iterate over the time bins
        for (time_bin in unique(df$grouped_time_bin)) {{
            # Subset the data for the current time bin
            sub_df <- df[df$grouped_time_bin == time_bin,]

            # Perform the interaction analysis
            interaction_model <- {interaction_model_formula}
            p_val_interact <- summary(interaction_model)${interaction_p_val_index}

            # Perform the within-category analyses
            p_val_PN <- perform_regression(sub_df[sub_df$category == 'PN', ], 'PN')
            p_val_IN <- perform_regression(sub_df[sub_df$category == 'IN', ], 'IN')

            # Store the results
            results <- rbind(results, data.frame(
                time_bin = time_bin,
                interaction_p_val = p_val_interact,
                PN_p_val = p_val_PN,
                IN_p_val = p_val_IN
            ))
        }}

        # Write the results to a CSV file
        write_csv(results, '{self.results_path}')
        '''

        self.script_path = os.path.join(self.data_opts['data_path'], self.data_type + '_r_script.r')
        with open(self.script_path, 'w') as f:
            f.write(r_script)

    def get_post_hoc_results(self, force_recalc=False):
        self.make_spreadsheet()
        self.results_path = os.path.join(self.data_opts['data_path'], 'r_results.csv')
        if not os.path.exists(self.results_path) or force_recalc:
            self.write_r_script()
            subprocess.run(['Rscript', self.script_path], check=True)
        results = pd.read_csv(self.results_path)
        interaction_p_vals = results['interaction_p_val'].tolist()
        within_neuron_type_p_vals = {nt: results[f'{nt}_p_val'].tolist() for nt in ('PN', 'IN')}
        return interaction_p_vals, within_neuron_type_p_vals

