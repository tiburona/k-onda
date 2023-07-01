import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects import DataFrame
import subprocess
import pandas as pd
import numpy as np
import csv
import os

from context import Base

pandas2ri.activate()
lme4 = importr('lme4')
base = importr('base')
lmerTest = importr('lmerTest')


class Stats(Base):
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

    def make_df(self):
        if self.data_opts['row_type'] == 'unit':
            rows = [row for unit in self.experiment.all_units for row in self.get_row(unit)]
        elif self.data_opts['row_type'] == 'trial':
            rows = [row for unit in self.experiment.all_units for trial in unit.trials for row in self.get_row(trial)]
        df = pd.DataFrame(rows)
        for var in ['unit_num', 'animal', 'category', 'condition']:
            df[var] = df[var].astype('category')
        self.df = df

    def make_spreadsheet(self):
        row_type = self.data_opts['row_type']
        path = self.data_opts.get('path')
        fname = os.path.join(path, '_'.join([self.data_type, self.time_type, row_type + 's']) + '.csv')
        self.spreadsheet_fname = fname

        with open(fname, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['unit_num', 'animal', 'condition', 'category', 'time_bin', self.data_col]
            if row_type == 'trial':
                header.insert(0, 'trial')
            writer.writerow(header)

            for row in self.get_rows():
                writer.writerow(row.values())

    def get_rows(self):
        if self.data_opts['row_type'] == 'unit':
            return [row for unit in self.experiment.all_units for row in self.get_row(unit)]
        elif self.data_opts['row_type'] == 'trial':
            return [row for unit in self.experiment.all_units for trial in unit.trials for row in self.get_row(trial)]

    def get_row(self, data_source):
        row_dict = self.get_row_dict(data_source)
        return getattr(self, f"{self.time_type}_rows")(data_source.data, row_dict)

    def get_row_dict(self, data_source):
        row_dict = {}
        if data_source.name == 'unit':
            unit = data_source
        elif data_source.name == 'trial':
            trial = data_source
            unit = trial.unit
            row_dict['trial'] = trial.identifier
        return {**row_dict,
                **{'unit_num': unit.identifier, 'animal': unit.animal.identifier, 'condition': unit.animal.condition,
                   'category': unit.neuron_type}}

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
        return [{**row_dict, **{'time_bin': time_bin, self.data_col: data[time_bin]}}
                for time_bin in range(self.num_time_points)]

    def get_post_hoc_results(self):
        if self.data_opts['post_hoc_type'] == 'beta':
            return self.beta_regression()
        else:
            return self.lmer_or_logit()

    def lmer_or_logit(self):
        self.make_df()
        divide_by = self.data_opts['post_hoc_bin_size']
        self.df['time_bin'] = self.df['time_point'].apply(lambda x: int(x / divide_by))
        time_bins = list(range(int(self.num_time_points / divide_by)))
        if divide_by == 1 or self.data_opts['row_type'] == 'unit':
            random = '(1|animal/unit_num)'
        else:
            random = '(1|animal/unit_num/trial)'
        interaction_p_vals = [self.interaction_p_val(self.df, time_bin, random) for time_bin in time_bins]
        within_neuron_type_p_vals = {
            neuron_type: [self.within_neuron_type_p_val(self.df.query(f"category == '{neuron_type}'"), time_bin, random)
                          for time_bin in time_bins]
            for neuron_type in ['PN', 'IN']}
        return interaction_p_vals, within_neuron_type_p_vals

    def interaction_p_val(self, df, time_bin, random):
        interaction_model = self.get_model(df, f"{self.data_col} ~ condition * category + {random}",
                                           time_bin)
        summary = base.summary(interaction_model)
        if self.data_opts['row_type'] == 'trial':
            return summary.rx('coefficients')[0][3][3]  # Adjust the indexing according to your summary structure
        else:
            return summary.rx('coefficients')[0][3][4]

    def within_neuron_type_p_val(self, df, time_bin, random):
        model = self.get_model(df, f"{self.data_col} ~ condition + {random}", time_bin)
        summary = base.summary(model)
        last_col = 3 if self.data_opts['row_type'] == 'trial' else 4
        return summary.rx('coefficients')[0][1][last_col]

    def get_model(self, df, model_def, time_bin):
        r_df = pandas2ri.py2rpy(df.query(f"time_bin == {time_bin}"))
        if self.data_opts['row_type'] == 'trial':
            return lme4.glmer(model_def, data=r_df, family=ro.r['binomial'](link="logit"))
        else:
            return lmerTest.lmer(model_def, data=r_df)

    def write_r_script(self):
        self.results_path = os.path.join(self.data_opts['path'], 'r_results.csv')
        r_script = fr'''
        library(glmmTMB)
        library(readr)

        df <- read_csv('{self.spreadsheet_fname}')

        # Convert variables to factors
        factor_vars <- c('unit_num', 'animal', 'category', 'condition')
        df[factor_vars] <- lapply(df[factor_vars], factor)

        # Add small constant to 0s in the data column
        df$"{self.data_col}"[df$"{self.data_col}" == 0] <- df$"{self.data_col}"[df$"{self.data_col}" == 0] + 1e-6

        # Create an empty data frame to store results
        results <- data.frame()

        perform_regression <- function(data, category){{
            # Perform beta regression with mixed effects
            mixed_beta <- glmmTMB(
                formula = {self.data_col} ~ condition + (1|animal/unit_num), 
                family = beta_family(link = "logit"), 
                data = data
            )

            # Extract p-value
            p_val <- summary(mixed_beta)$coefficients$cond[2, 4]
            return(p_val)
        }}

        # Iterate over the time bins
        for (time_bin in unique(df$time_bin)) {{
            # Subset the data for the current time bin
            sub_df <- df[df$time_bin == time_bin,]

            # Perform the interaction analysis
            interaction_model <- glmmTMB(
                formula = {self.data_col} ~ condition * category + (1|animal/unit_num),
                family = beta_family(link = "logit"),
                data = sub_df
            )
            p_val_interact <- summary(interaction_model)$coefficients$cond["conditionstressed:categoryPN", 4]

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

        self.script_path = os.path.join(self.data_opts['path'], 'r_script.r')
        with open(self.script_path, 'w') as f:
            f.write(r_script)

    def execute_r_script(self):
        subprocess.run(['Rscript', self.script_path], check=True)
        results = pd.read_csv(self.results_path)
        interaction_p_vals = results['interaction_p_val'].tolist()
        within_neuron_type_p_vals = {nt: results[f'{nt}_p_val'].tolist() for nt in ('PN', 'IN')}
        return interaction_p_vals, within_neuron_type_p_vals

    def beta_regression(self):
        self.make_spreadsheet()
        self.write_r_script()
        return self.execute_r_script()



    # def get_animal_rows(self, animal):
    #     rows = []
    #     for neuron_type in ['PN', 'IN']:
    #         animal.selected_neuron_type = neuron_type
    #         row_dict = {'animal': animal, 'condition': animal.group.identifier, 'category': neuron_type}
    #         rows.append(getattr(self, f"{self.time_type}_rows")(animal.data, row_dict))
    #     return rows
