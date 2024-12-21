import subprocess
import pandas as pd
import numpy as np
import csv
import os

from context import Base


class Stats(Base):
    """A class to construct dataframes, write out csv files, and call R for statistical tests."""
    def __init__(self, experiment, data_type_context, data_opts):
        self.experiment = experiment
        self.data_type_context = data_type_context
        self.data_opts = data_opts
        self.df = None
        self.time_type = self.data_opts['time']
        self.time_col = 'time_point' if self.time_type == 'continuous' else 'period'
        self.data_col = 'rate' if self.data_type == 'psth' else 'proportion'
        pre_stim, post_stim, bin_size = (self.data_opts[k] for k in ('pre_stim', 'post_stim', 'bin_size'))
        self.num_time_points = int((pre_stim + post_stim)/bin_size)
        self.spreadsheet_fname = None
        self.bootstrap_sems = None
        self.results_path = None
        self.script_path = None

    def make_df(self):
        rows = self.get_rows()
        df = pd.DataFrame(rows)
        for var in ['unit_num', 'animal', 'category', 'condition', 'two_way_split', 'three_way_split']:
            df[var] = df[var].astype('category')
        self.df = df

    def make_spreadsheet(self):
        row_type = self.data_opts['row_type']
        path = self.data_opts.get('path')
        fname = os.path.join(path, '_'.join([self.data_type, self.time_type, row_type + 's']) + '.csv')
        self.spreadsheet_fname = fname

        header = ['unit_num', 'animal', 'condition', 'category', 'two_way_split', 'three_way_split', 'time_bin',
                  'grouped_time_bin', self.data_col]
        if row_type == 'trial':
            header.insert(0, 'trial')

        with open(fname, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()

            for row in self.get_rows():
                writer.writerow(row)

    def get_rows(self):
        if self.data_opts['row_type'] == 'unit':
            return [row for unit in self.experiment.all_units for row in self.get_row(unit)]
        elif self.data_opts['row_type'] == 'trial':
            return [row for unit in self.experiment.all_units for trial in unit.trials for row in self.get_row(trial)]

    def get_row(self, data_source):
        row_dict = self.get_row_dict(data_source)
        return getattr(self, f"{self.time_type}_rows")(data_source.data, row_dict)

    @staticmethod
    def get_row_dict(data_source):
        row_dict = {}

        if data_source.name == 'unit':
            unit = data_source
        elif data_source.name == 'trial':
            trial = data_source
            unit = trial.unit
            row_dict['trial'] = trial.identifier

        row_dict = {**row_dict,
                    **{'unit_num': unit.identifier, 'animal': unit.animal.identifier,
                       'condition': unit.animal.condition, 'category': unit.neuron_type}}
        return row_dict

    @staticmethod
    def two_way_split(time_bin):
        return 'early' if time_bin < 30 else 'late'

    @staticmethod
    def three_way_split(time_bin):
        if time_bin < 5:
            return 'pip'
        elif time_bin < 30:
            return 'early'
        else:
            return 'late'

    @staticmethod
    def calculate_rate(data, bin_slice):
        return np.mean(data[slice(*bin_slice)])

    def binned_rows(self, data, row_dict):
        periods = ['during_beep', 'early_post_beep', 'mid_post_beep', 'late_post_beep']
        bin_slices = [(0, 5), (5, 30), (30, 60), (60, 100)]
        rows = []
        for period, bin_slice in zip(periods, bin_slices):
            rate = self.calculate_rate(data, bin_slice)
            rows.append({**row_dict, **{'period': period, 'rate': rate}})
        return rows

    def continuous_rows(self, data, row_dict):
        post_hoc_bin_size = self.data_opts.get('post_hoc_bin_size', 1)  # Default to 1 if not present
        return [{**row_dict,
                 **{'two_way_split': self.two_way_split(time_bin),
                    'three_way_split': self.three_way_split(time_bin),
                    'time_bin': time_bin,
                    'grouped_time_bin': time_bin // post_hoc_bin_size,  # Calculate 'grouped_time_bin'
                    self.data_col: data[time_bin]}}
                for time_bin in range(self.num_time_points)]

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

        self.results_path = os.path.join(self.data_opts['path'], 'r_results.csv')
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

        self.script_path = os.path.join(self.data_opts['path'], self.data_type + '_r_script.r')
        with open(self.script_path, 'w') as f:
            f.write(r_script)

        self.script_path = os.path.join(self.data_opts['path'], self.data_type + '_r_script.r')
        with open(self.script_path, 'w') as f:
            f.write(r_script)

    def execute_r_script(self):
        subprocess.run(['Rscript', self.script_path], check=True)
        results = pd.read_csv(self.results_path)
        interaction_p_vals = results['interaction_p_val'].tolist()
        within_neuron_type_p_vals = {nt: results[f'{nt}_p_val'].tolist() for nt in ('PN', 'IN')}
        return interaction_p_vals, within_neuron_type_p_vals

    def get_post_hoc_results(self):
        self.make_spreadsheet()
        self.write_r_script()
        return self.execute_r_script()
