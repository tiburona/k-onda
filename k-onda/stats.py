import subprocess
import pandas as pd
import numpy as np
import csv
import os

from context import Base
from utils import range_args
from lfp import FrequencyPeriod
from spike_data import *

LEVEL_DICT = dict(animal=Animal, group=Group, unit=Unit, period=Period, frequency_period=FrequencyPeriod)


class Stats(Base):
    """A class to construct dataframes, write out csv files, and call R for statistical tests."""
    def __init__(self, experiment, data_type_context, neuron_type_context, data_opts):
        self.experiment = experiment
        self.data_type_context = data_type_context
        self.data_opts = data_opts
        self.neuron_type_context = neuron_type_context
        self.selected_neuron_type = None
        self.dfs = {}
        self.time_type = self.data_opts['time']
        self.data_col = None
        self.num_time_points = None
        self.spreadsheet_fname = None
        self.results_path = None
        self.script_path = None
        self.lfp_brain_region = None
        self.frequency_band = None

    @property
    def rows(self):
        return self.get_rows()

    def set_attributes(self):
        if self.data_type == 'lfp':
            self.data_col = 'power'
            self.frequency_band = self.data_opts['fb']
            self.lfp_brain_region = self.data_opts['brain_region']
        else:
            self.data_col = 'rate' if self.data_type == 'psth' else 'proportion'
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
            result = pd.merge(result, df, how='left', on=keys)

        new_df_name = '_'.join(df_names)
        self.dfs[new_df_name] = result
        return new_df_name

    def make_dfs(self, names, opts_dicts):
        for name, opts in zip(names, opts_dicts):
            self.data_opts = opts
            self.make_df(name)
        common_columns = set(self.dfs[names[0]].columns)
        for df in self.dfs.values():
            common_columns &= set(df.columns)
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
        if 'lfp' in self.data_type:
            [animal.get_lfp() for animal in Animal.instances]
        else:
            [animal.update(self.neuron_type_context) for animal in Animal.instances]
            LEVEL_DICT[self.data_opts['row_type']].initialize_data()  # TimeBins and Periods aren't created automatically

    def get_rows(self, inclusion_criteria=None, other_attributes=None):
        if other_attributes is None:
            other_attributes = []
        if inclusion_criteria is None:
            inclusion_criteria = []
        if 'lfp' in self.data_type:
            level = FrequencyPeriod
        else:
            level = TimeBin if self.data_opts['time'] == 'continuous' else Period
            other_attributes.append(lambda x: ('category', x.category) if x.name == 'unit' else None)
            parent = LEVEL_DICT[self.data_opts['row_type']]
            inclusion_criteria.append(lambda x: isinstance(x.parent, parent))
        other_attributes.append(lambda x: ('period_type', x.period_type) if x.name == 'period' else None)

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
                for other_attribute in other_attributes:
                    attribute = other_attribute(current_instance)
                    if attribute is not None:
                        attribute_name, attribute_val = attribute
                        row_dict[attribute_name] = attribute_val  # can use this for stage
                if hasattr(current_instance, 'parent') and current_instance.parent is not None:
                    current_instance = current_instance.parent  # Go one level up to the parent
                else:
                    break  # Stop the loop if there is no parent
            rows.append(row_dict)
        return rows

    def make_spreadsheet(self, df_name=None):
        row_type = self.data_opts['row_type']
        path = self.data_opts.get('data_path')
        name = df_name if df_name else self.data_type
        if 'lfp' in name:
            path = os.path.join(path, 'lfp')
            name += f"_{self.lfp_brain_region}_{self.frequency_band}"
        fname = os.path.join(path, '_'.join([name, self.time_type, row_type + 's']) + '.csv')
        self.spreadsheet_fname = fname
        self.make_df(df_name)

        with open(fname, 'w', newline='') as f:
            header = list()  # fig
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

