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
        self.dfs = {}
        self.time_type = self.data_opts['time']
        self.data_col = None
        self.num_time_points = None
        self.spreadsheet_fname = None
        self.results_path = None
        self.script_path = None
        self.lfp_brain_region = None
        self.frequency_band = None

    def set_properties(self):
        if self.data_type == 'lfp':
            self.data_col = 'power'
            self.frequency_band = self.data_opts['fb']
            self.lfp_brain_region = self.data_opts['brain_region']
        else:
            self.data_col = 'rate' if self.data_type == 'psth' else 'proportion'
            pre_stim, post_stim, bin_size = (self.data_opts[k] for k in ('pre_stim', 'post_stim', 'bin_size'))
            self.num_time_points = int((pre_stim + post_stim) / bin_size)

    def make_df(self, name):
        self.set_properties()
        rows = self.get_rows()
        df = pd.DataFrame(rows)
        vs = ['unit_num', 'animal', 'category', 'condition', 'two_way_split', 'three_way_split', 'frequency']
        for var in vs:
            if var in df:
                df[var] = df[var].astype('category')
        self.dfs[name] = df

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
        new_name = self.smart_merge(('animal', 'condition', 'period'))
        return new_name

    def get_lfp_rows(self):
        animals = [animal for group in self.experiment.groups for animal in group.children]
        rows = []
        for animal in animals:
            lfp = animal.get_lfp()
            for i, period_val in enumerate(lfp.normalized_power[self.data_opts['fb']]):
                rows.append({'condition': animal.condition, 'animal': animal.identifier, 'period': i,
                            'power': period_val})
        return rows

    def make_spreadsheet(self, df_name=None):
        row_type = self.data_opts['row_type']
        path = self.data_opts.get('path')
        name = df_name if df_name else self.data_type
        if 'lfp' in name:
            path = os.path.join(path, 'lfp')
            name += f"_{self.lfp_brain_region}_{self.frequency_band}"
        fname = os.path.join(path, '_'.join([name, self.time_type, row_type + 's']) + '.csv')
        self.spreadsheet_fname = fname

        with open(fname, 'w', newline='') as f:
            header = self.get_header(df_name=df_name)
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()

            if df_name is None:
                for row in self.get_rows():
                    writer.writerow(row)
            else:
                for index, row in self.dfs[df_name].iterrows():
                    writer.writerow(row.to_dict())

    def get_header(self, df_name=None):
        if df_name is None:
            df_name = self.data_type
        row_type = self.data_opts.get('row_type')
        header = ['animal', 'condition', 'category']
        if 'psth' in df_name or 'proportion' in df_name:
            header += ['unit_num']
        if 'lfp' not in df_name:
            header += ['two_way_split', 'three_way_split']
        if row_type == 'trial':
            header += ['trial']
        if self.data_opts['time'] == 'period':
            header += ['period']
        else:
            header += ['grouped_time_bin', 'time_bin']
        if df_name:
            data_cols = dict(proportion='proportion', psth='rate', lfp='power')
            dfs = df_name.split('_')
            for df in dfs:
                header += [data_cols[df]]
        else:
            header += [self.data_col]
        return header

    def get_rows(self):
        if self.data_type == 'lfp':
            return self.get_lfp_rows()
        else:
            if self.data_opts['row_type'] == 'unit':
                return [row for unit in self.experiment.all_units for row in self.get_row(unit)]
            elif self.data_opts['row_type'] == 'trial':
                return [row for unit in self.experiment.all_units for trial in unit.trials for row in self.get_row(trial)]

    def get_row(self, data_source):
        row_dict = self.get_row_dict(data_source)
        return getattr(self, f"{self.time_type}_rows")(data_source, row_dict)

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

    def binned_rows(self, data_source, row_dict):
        data = data_source.data
        periods = ['during_beep', 'early_post_beep', 'mid_post_beep', 'late_post_beep']
        bin_slices = [(0, 5), (5, 30), (30, 60), (60, 100)]
        rows = []
        for period, bin_slice in zip(periods, bin_slices):
            rate = self.calculate_rate(data, bin_slice)
            rows.append({**row_dict, **{'period': period, 'rate': rate}})
        return rows

    def continuous_rows(self, data_source, row_dict):
        data = data_source.data
        post_hoc_bin_size = self.data_opts.get('post_hoc_bin_size', 1)  # Default to 1 if not present
        return [{**row_dict,
                 **{'two_way_split': self.two_way_split(time_bin),
                    'three_way_split': self.three_way_split(time_bin),
                    'time_bin': time_bin,
                    'grouped_time_bin': time_bin // post_hoc_bin_size,  # Calculate 'grouped_time_bin'
                    self.data_col: data[time_bin]}}
                for time_bin in range(self.num_time_points)]

    def period_rows(self, data_source, row_dict):
        data = data_source.mean_over_period()
        return [{**row_dict, **{'period': i, self.data_col: data[i]}} for i, period in enumerate(data)]

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

        self.script_path = os.path.join(self.data_opts['path'], self.data_type + '_r_script.r')
        with open(self.script_path, 'w') as f:
            f.write(r_script)

    def get_post_hoc_results(self, force_recalc=False):
        self.make_spreadsheet()
        self.results_path = os.path.join(self.data_opts['path'], 'r_results.csv')
        if not os.path.exists(self.results_path) or force_recalc:
            self.write_r_script()
            subprocess.run(['Rscript', self.script_path], check=True)
        results = pd.read_csv(self.results_path)
        interaction_p_vals = results['interaction_p_val'].tolist()
        within_neuron_type_p_vals = {nt: results[f'{nt}_p_val'].tolist() for nt in ('PN', 'IN')}
        return interaction_p_vals, within_neuron_type_p_vals

