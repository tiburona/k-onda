import subprocess
import pandas as pd
import csv
import os
from copy import deepcopy
import random
import string
from data import Base
from functools import reduce
from collections import namedtuple


class Stats(Base):
    """A class to construct dataframes, write out csv files, and call R for statistical tests."""
    def __init__(self, experiment, lfp=None, behavior=None):
        self.experiment = experiment
        self.lfp = lfp
        self.behavior = behavior
        self.dfs = {}
        self.data_col = None
        self.spreadsheet_fname = None
        self.results_path = None
        self.script_path = None
        self.opts_dicts = []
        self.name_suffix = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(6))

    def set_attributes(self, data_opts):
        self.data_opts = data_opts
        if self.data_class == 'lfp':
            fb = self.current_frequency_band
            if not isinstance(self.current_frequency_band, str):
                translation_table = str.maketrans({k: '_' for k in '[](),'})
                fb = str(list(fb)).translate(translation_table)
            if any([s in self.data_type for s in ['coherence', 'correlation', 'phase']]):
                self.data_col = f"{self.data_opts['region_set']}_{fb}_{self.data_type}"
            else:
                self.data_col = f"{self.current_brain_region}_{fb}_{self.data_type}"
        else:
            self.data_col = 'rate' if self.data_type == 'psth' else self.data_type

    def make_df(self, data_opts):
        self.set_attributes(data_opts)
        self.opts_dicts.append(deepcopy(self.data_opts))
        name = self.set_df_name()
        df = pd.DataFrame(self.get_rows())
        vs = ['unit_num', 'animal', 'category', 'group', 'frequency']
        for var in vs:
            if var in df:
                df[var] = df[var].astype('category')
        if name in self.dfs:
            name += '_2'
        self.dfs[name] = df

    def set_df_name(self):
        name = self.data_type
        if 'lfp' in self.data_class:
            name += f"_{self.current_brain_region}_{self.current_frequency_band}"
        return name

    def merge_dfs_animal_by_animal(self):


        df_items = list(self.dfs.items())
        names, dfs = zip(*df_items)

        # Extract unique animals across all data frames
        unique_animals = pd.concat([df['animal'] for df in dfs if 'animal' in df.columns]).unique()

        # List to hold merged data for each animal
        merged_data_per_animal = []

        for animal in unique_animals:
            # Filter DataFrames for the current animal, remove 'animal' column
            dfs_per_animal = [df[df['animal'] == animal].copy().drop(columns=['animal']) for df in dfs if animal in df['animal'].values]

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
        new_df_name = '_'.join([name for name, _ in df_items])
        self.dfs[new_df_name] = final_merged_df
        return new_df_name



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
        level = self.data_opts['row_type']
        
        other_attributes = ['period_type']
        
        if 'lfp' in self.data_class:
            if self.data_type in ['mrl']:
                level = 'mrl_calculator'
                other_attributes += ['frequency', 'fb', 'neuron_type', 'neuron_quality']  # TODO: figure out what fb should be changed to post refactor
            if any([w in self.data_type for w in ['coherence', 'correlation', 'phase']]):
                other_attributes.append('period_id')
            else:
                if self.data_opts['time_type'] == 'continuous' and self.data_opts.get('power_deviation'):
                    other_attributes.append('power_deviation')
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

        sources = [source for source in getattr(experiment, f'all_{level}s') if source.is_valid]

        if self.data_opts.get('frequency_type') == 'continuous':
            other_attributes.append('frequency')
            sources = [frequency_bin for source in sources for frequency_bin in source.frequency_bins]
        if self.data_opts.get('time_type') == 'continuous':
            other_attributes.append('time')
            sources = [time_bin for source in sources for time_bin in source.time_bins]

        for source in sources:

            if self.data_opts.get('aggregator') == 'sum':
                row_dict = {self.data_col: source.sum_data}
            elif self.data_opts.get('aggregator') == 'none':
                if source.key:
                    row_dict = {f"{self.data_col}_{source.key}": source.data}
                else:
                    row_dict = {self.data_col: source.data}
            else:
                row_dict = {self.data_col: source.mean_data}
            
            for src in source.ancestors:
                row_dict[src.name] = src.identifier
                for attr in other_attributes:
                    val = getattr(src, attr) if hasattr(src, attr) else None
                    if val is not None:
                        if attr == 'power_deviation':
                            attr = f"{self.lfp.brain_region}_{self.current_frequency_band}_{attr}"
                        row_dict[attr] = val
            rows.append(row_dict)

        return rows

    def make_spreadsheet(self, path=None, filename=None, force_recalc=True):
        """
        Creates a spreadsheet (CSV file) from a specified DataFrame stored within the object.

        This function constructs the filename based on various attributes and options set in the objects `data_opts`. If
        the file already exists and `force_recalc` is set to False, the function will not overwrite the existing file.

        Parameters:
        - path (str, optional): Directory path where the spreadsheet should be saved. If not provided, it defaults to
          the `data_path` attribute from the object's `data_opts`.
        - filename (str, optional): Name of the CSV file.  Will be generated automatically if absent.
        - force_recalc (bool, optional): If set to True, the function will overwrite an existing file with the same
          name. Defaults to True.

        Returns:
        None: The function saves the spreadsheet to the specified path and updates the `spreadsheet_fname` attribute
           of the object.
        """
        if len(self.dfs):
            df_name = self.merge_dfs_animal_by_animal()
        else:
            self.make_df(self.data_opts)
            df_name = self.data_type
        if path is None:
            path = self.data_opts['data_path']
        path = os.path.join(path, self.data_type)
        if not os.path.exists(path):
            os.mkdir(path)
        if filename is None:
            filename = df_name
        self.spreadsheet_fname = os.path.join(path, filename + '.csv')
        if os.path.exists(self.spreadsheet_fname) and not force_recalc:
            return
        try:
            with open(self.spreadsheet_fname, 'w', newline='') as f:
                self.write_csv(f, df_name)
        except OSError:  # automatically generated name is too long
            self.spreadsheet_fname = self.spreadsheet_fname[0:75] + self.name_suffix
            self.spreadsheet_fname += self.name_suffix
            self.spreadsheet_fname += '.csv'
            with open(self.spreadsheet_fname, 'w', newline='') as f:
                self.write_csv(f, df_name)


    def write_csv(self, f, df_name):
        for opts_dict in self.opts_dicts:
            line = ', '.join([f"{str(key).replace(',', '_')}: {str(value).replace(',', '_')}" for key, value in
                              opts_dict.items()])
            f.write(f"# {line}\n")
            f.write("\n")

        header = list(self.dfs[df_name].columns)
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for index, row in self.dfs[df_name].iterrows():
            writer.writerow(row.to_dict())

    def write_post_hoc_r_script(self):

        r_script = self.write_spike_post_hoc_r_script()

        self.script_path = os.path.join(self.data_opts['data_path'], self.data_type, 'post_hoc', self.data_type +
                                        '_r_script.r')
        with open(self.script_path, 'w') as f:
            f.write(r_script)

    def write_spike_post_hoc_r_script(self):
         
        full_error_suffixes = {'event': '/unit/period',  'period': '/unit'}

        error_suff_dict = {
            'beta': full_error_suffixes,
            'lmer': full_error_suffixes,
            'poisson': {
                'event': ':unit) + (1|animal:unit:period)',
                'period': ':unit)'
            }
        }
         
        error_suffix = error_suff_dict[self.data_opts['post_hoc_type']][self.data_opts['row_type']]

        if self.data_opts.get('post_hoc_bin_size', 1) > 1:
            if self.data_opts['post_hoc_type'] == 'poisson':
                error_suffix += '(1|animal:unit:period:time_bin)'
            else:
                error_suffix += '/time_bin' 

        post_hoc_bin_size = self.data_opts.get('post_hoc_bin_size', 1)
        
        if self.data_opts.get('period_type_regressor'):
            pt_regressor_str = '* period_type'
            within_nt_p_row = 4
            interaction_p_row = 8
        else:
            pt_regressor_str = ''
            within_nt_p_row = 2
            interaction_p_row = 4


        if self.data_opts['post_hoc_type'] == 'beta':
            model_formula = f'glmmTMB(formula = {self.data_col} ~ group {pt_regressor_str} + (1|animal{error_suffix}), ' \
                            f'family = beta_family(link = "logit"), data = data)'
            interaction_model_formula = f'glmmTMB(formula = {self.data_col} ~ group * neuron_type {pt_regressor_str} + ' \
                                        f'(1|animal{error_suffix}), ' \
                                        f'family = beta_family(link = "logit"), data = sub_df)'
            p_val_index = f'coefficients$cond[{within_nt_p_row}, 4]'
            interaction_p_val_index = f'coefficients$cond[{interaction_p_row}, 4]'
            zero_adjustment_line = f'df$"{self.data_col}"[df$"{self.data_col}" == 0] ' \
                                   f'<- df$"{self.data_col}"[df$"{self.data_col}" == 0] + 1e-6'
        elif self.data_opts['post_hoc_type'] == 'lmer':
            model_formula = f'lmer({self.data_col} ~ group {pt_regressor_str} + (1|animal{error_suffix}), data = data)'
            interaction_model_formula = f'lmer({self.data_col} ~ group * neuron_type {pt_regressor_str} + (1|animal{error_suffix}), data = sub_df)'
            p_val_index = f'coefficients[{within_nt_p_row}, 5]'
            interaction_p_val_index = f'coefficients[{interaction_p_row}, 5]'
            zero_adjustment_line = ''
        elif self.data_opts['post_hoc_type'] == 'poisson':
            model_formula = f'glmmTMB({self.data_col} ~ group {pt_regressor_str} + (1|animal:unit) + (1|animal:unit:period), ' \
            'ziformula = ~ 1, family = poisson(link = "log"), data = data)'
            interaction_model_formula = f'glmmTMB({self.data_col} ~ group * neuron_type {pt_regressor_str}  + (1|animal{error_suffix}, data = sub_df)' 
            p_val_index = f'coefficients$cond[{within_nt_p_row}, 4]'
            interaction_p_val_index = f'coefficients$cond[{interaction_p_row}, 4]'
            zero_adjustment_line = ''
        else:
            raise ValueError('Unknown post hoc type')

        select_period_type_line = ''
        if self.experiment.hierarchy[self.data_opts['row_type']] >  3 and not self.data_opts.get('period_type_regressor'):
            select_period_type_line = "df <- df[df$period_type == 'tone', ]"  # TODO make this not specific to tone

        return fr'''
                library(glmmTMB)
                library(lme4)
                library(lmerTest)  # for lmer p-values
                library(readr)

                df <- read.csv('{self.spreadsheet_fname}', comment.char="#")

                # Convert variables to factors
                factor_vars <- c('unit', 'animal', 'neuron_type', 'group')
                df[factor_vars] <- lapply(df[factor_vars], factor)

                {select_period_type_line}

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

                 unique_time_bins <- sort(unique(df$time_bin))
                
                # Calculate the total number of groups needed
                total_groups <- ceiling(length(unique_time_bins) / {post_hoc_bin_size})

                # Iterate over the time bins
                for (group_index in 1:total_groups) {{
                    start_index <- (group_index - 1) * {post_hoc_bin_size} + 1
                    end_index <- min(length(unique_time_bins), group_index * {post_hoc_bin_size})
                    # Get the time_bin values for the current group
                    current_group_bins <- unique_time_bins[start_index:end_index]
                    # Subset the data for the current time bin
                    sub_df <- df[df$time_bin %in% current_group_bins, ]

                    if (nrow(sub_df) > 0) {{

                    # Perform the interaction analysis
                    interaction_model <- {interaction_model_formula}
                    p_val_interact <- summary(interaction_model)${interaction_p_val_index}

                    # Perform the within-category analyses
                    p_val_PN <- perform_regression(sub_df[sub_df$neuron_type == 'PN', ], 'PN')
                    p_val_IN <- perform_regression(sub_df[sub_df$neuron_type == 'IN', ], 'IN')

                    # Store the results
                    results <- rbind(results, data.frame(
                        time_bin = group_index,
                        interaction_p_val = p_val_interact,
                        PN_p_val = p_val_PN,
                        IN_p_val = p_val_IN
                    ))
                    }}
                }}

                # Write the results to a CSV file
                write_csv(results, '{self.results_path}')
                '''

    def get_post_hoc_results(self, force_recalc=True):
        post_hoc_path = os.path.join(self.data_opts['data_path'], self.data_type, 'post_hoc')
        if not os.path.exists(post_hoc_path):
            os.mkdir(post_hoc_path)
        adjustment = self.data_opts.get('adjustment')
        changes_to_data_opts = []
        if self.data_opts.get('period_type_regressor'):
            changes_to_data_opts.append((['adjustment'], 'none'))
        if self.data_opts.get('post_hoc_type') == 'poisson':
            changes_to_data_opts.append((['data_type'], 'spike_counts'))
        self.update_data_opts(changes_to_data_opts)
        self.make_spreadsheet(path=post_hoc_path)
        self.results_path = os.path.join(post_hoc_path, 'r_post_hoc_results.csv')
        if not os.path.exists(self.results_path) or force_recalc:
            getattr(self, f"write_post_hoc_r_script")()
            env = os.environ.copy()
            env['FLIBS'] = '-L/usr/local/lib/gcc/13 -lgfortran -lquadmath -lm'
            result = subprocess.run(['/Library/Frameworks/R.framework/Resources/bin/Rscript', 
                                     self.script_path], capture_output=True, text=True)
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
        results = pd.read_csv(self.results_path)
        if self.data_opts.get('period_type_regressor'):
            self.update_data_opts([(['adjustment'], adjustment)])
        return getattr(self, f"construct_{self.data_class}_post_hoc_results")(results)

    def construct_spike_post_hoc_results(self, results):
        interaction_p_vals = results['interaction_p_val'].tolist()
        within_neuron_type_p_vals = {nt: results[f'{nt}_p_val'].tolist() for nt in ('PN', 'IN')}
        return interaction_p_vals, within_neuron_type_p_vals

    def construct_lfp_post_hoc_results(self, results):
        return results

    def write_lfp_post_hoc_r_script(self):  # TODO: this should really be moved to a separate R interface class so that brittle R methods are isolated
        # TODO: why, below, are frequency band and brain region evoked with current and without on separate lines.
        prepare_df_script = f"""
        prepare_df <- function(frequency_band, brain_region, evoked=FALSE) {{
            csv_name = sprintf('mrl_%s_continuous_period_frequency_bins_wavelet_%s.csv', frequency_band, brain_region)
            csv_file = paste('{os.path.join(self.data_path, 'lfp', 'mrl')}', csv_name, sep='/')
            df <- read.csv(csv_file, comment.char="#")

            factor_vars <- c('animal', 'group', 'period_type', 'neuron_type', 'unit')
            df[factor_vars] <- lapply(df[factor_vars], factor)

            mrl_column <- sym(paste0(frequency_band, "_mrl"))

            averaged_over_unit_result <- df %>%
                group_by(animal, period_type, period, neuron_type, group, frequency_bin) %>%
                summarize(
                    mean_mrl = mean(!!mrl_column, na.rm = TRUE)
                ) %>%
                ungroup()

            data_to_return <- averaged_over_unit_result

            if (evoked) {{
                pretone_data <- subset(averaged_over_unit_result, period_type == "pretone")
                tone_data <- subset(averaged_over_unit_result, period_type == "tone")

                merged_data <- merge(pretone_data, tone_data, by = setdiff(names(averaged_over_unit_result), c("mean_mrl", "period_type")))
                merged_data$mrl_diff <- merged_data$mean_mrl.x - merged_data$mean_mrl.y
                data_to_return <- merged_data
            }}

            return(data_to_return)
        }}
        """

        regression_script = f"""
        perform_tests <- function(df, evoked=FALSE) {{
            results <- data.frame()

            # ... Rest of the regression script ...

            return(results)
        }}

        data <- prepare_df('{self.current_frequency_band}', '{self.lfp.brain_region}')  
        evoked_data <- prepare_df('{self.frequency_band}', '{self.brain_region}', evoked=TRUE)

        results <- rbind(perform_tests(data), perform_tests(evoked_data, evoked=TRUE))

        write_csv(results, '{self.results_path}')
        """

        complete_script = f"""
        library(glmmTMB)
        library(lme4)
        library(lmerTest)
        library(readr)
        library(dplyr)
        library(tidyr)

        {prepare_df_script}
        {self.mrl_post_hoc_regression_script()}
        """

        return complete_script

    def mrl_post_hoc_regression_script(self):
        return f"""
        perform_tests <- function(df, evoked=FALSE) {{
            results <- data.frame()

            groups <- c('control', 'defeat')
            neuron_types <- c('IN', 'PN')

            if (!evoked) {{

                # mrl ~ period_type + period + (1|animal) within each combination of control/defeat and IN/PN
                for (group in groups) {{
                    for (neuron in neuron_types) {{
                        sub_df <- df[df$group == group & df$neuron_type == neuron,]
                        model <- lmer(mean_mrl ~ period_type + period + (1|animal), data=sub_df)
                        p_vals <- summary(model)$coefficients[,5]  # Assuming 5th column has p-values
                        non_intercept_pvals <- p_vals[names(p_vals) != '(Intercept)']
                        results <- rbind(results, data.frame(test='within-group-and-neuron', group=group, neuron_type=neuron, p_values=paste(non_intercept_pvals, collapse=',')))
                    }}
                }}

                # mrl ~ group*period_type + period + (1/animal) in each subset of neuron_type
                for (neuron in neuron_types) {{
                    sub_df <- df[df$neuron_type == neuron,]
                    model <- lmer(mean_mrl ~ group*period_type + period + (1|animal), data=sub_df)
                    p_vals <- summary(model)$coefficients[,5]
                    non_intercept_pvals <- p_vals[names(p_vals) != '(Intercept)']
                    results <- rbind(results, data.frame(test='interaction-with-group', neuron_type=neuron, p_values=paste(non_intercept_pvals, collapse=',')))
                }}

                # mrl ~ neuron_type*period_type + period + (1/animal) in each subset of group
                for (group in groups) {{
                    sub_df <- df[df$group == group,]
                    model <- lmer(mean_mrl ~ neuron_type*period_type + period + (1|animal), data=sub_df)
                    p_vals <- summary(model)$coefficients[,5]
                    non_intercept_pvals <- p_vals[names(p_vals) != '(Intercept)']
                    results <- rbind(results, data.frame(test='interaction-with-neuron', group=group, p_values=paste(non_intercept_pvals, collapse=',')))
                }}

            }} else {{

                # For evoked data

                # evoked_mrl ~ group + period + (1|animal) within subsets of IN and PN
                for (neuron in neuron_types) {{
                    sub_df <- df[df$neuron_type == neuron,]
                    model <- lmer(mrl_diff ~ group + period + (1|animal), data=sub_df)
                    p_vals <- summary(model)$coefficients[,5]
                    non_intercept_pvals <- p_vals[names(p_vals) != '(Intercept)']
                    results <- rbind(results, data.frame(test='evoked-within-neuron', neuron_type=neuron, p_values=paste(non_intercept_pvals, collapse=',')))
                }}

                # evoked_mrl ~ neuron_type + period + (1/animal) within control/defeat
                for (group in groups) {{
                    sub_df <- df[df$group == group,]
                    model <- lmer(mrl_diff ~ neuron_type + period + (1|animal), data=sub_df)
                    p_vals <- summary(model)$coefficients[,5]
                    non_intercept_pvals <- p_vals[names(p_vals) != '(Intercept)']
                    results <- rbind(results, data.frame(test='evoked-within-group', group=group, p_values=paste(non_intercept_pvals, collapse=',')))
                }}

            }}

            return(results)
        }}
        """




