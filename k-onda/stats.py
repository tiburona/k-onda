import subprocess
import pandas as pd
import csv
import os
from copy import deepcopy
import random
import string
from data import Base
from utils import find_ancestor_attribute, find_ancestor_id


class Stats(Base):
    """A class to construct dataframes, write out csv files, and call R for statistical tests."""
    def __init__(self, experiment, data_opts, lfp=None, behavior=None):
        self.experiment = experiment
        self.lfp = lfp
        self.behavior = behavior
        self.data_opts = data_opts
        self.dfs = {}
        self.data_col = None
        self.spreadsheet_fname = None
        self.results_path = None
        self.script_path = None
        self.opts_dicts = []
        self.name_suffix = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(6))

    @property
    def rows(self):
        return self.get_rows()

    def set_attributes(self, brain_region=None, frequency_band=None):
        if self.data_class == 'lfp':
            self.current_brain_region = brain_region
            self.current_frequency_band = frequency_band
            self.data_col = f"{self.current_brain_region}_{self.current_frequency_band}_{self.data_type}"
        else:
            self.data_col = 'rate' if self.data_type == 'psth' else self.data_type

    def make_dfs(self, opts_dicts):
        """
         Constructs one or more DataFrames based on the provided options dictionaries.

         The function iterates over each options dictionary in `opts_dicts`, sets the object's `data_opts` attribute,
         and then calls the `self.make_df` method to create a DataFrame. If the data class includes 'lfp', the function
         creates a DataFrame for each frequency band specified in the options. Otherwise, it creates a single DataFrame
         based on the object's data type. After constructing all DataFrames, it calls `self.merge_dfs` to merge them.

         Parameters:
         - opts_dicts (list of dict): A list of dictionaries, where each dictionary contains options that influence how
           the DataFrame is constructed.

         Returns:
         DataFrame: A merged DataFrame constructed based on the provided options dictionaries and the object's attributes.
         """
        for opts in opts_dicts:
            self.data_opts = opts
            if 'lfp' in self.data_class:
                for brain_region in self.data_opts['brain_regions']:
                    for frequency_band in self.data_opts['frequency_bands']:
                        self.make_df(brain_region=brain_region, frequency_band=frequency_band)
            else:
                self.make_df()
        return self.merge_dfs()

    def make_df(self, brain_region=None, frequency_band=None):
        self.set_attributes(brain_region=brain_region, frequency_band=frequency_band)
        self.opts_dicts.append(deepcopy(self.data_opts))
        name = self.set_df_name()
        df = pd.DataFrame(self.rows)
        vs = ['unit_num', 'animal', 'category', 'group', 'two_way_split', 'three_way_split', 'frequency']
        for var in vs:
            if var in df:
                df[var] = df[var].astype('category')
        if name in self.dfs:
            name += '_2'
        self.dfs[name] = df

    def set_df_name(self):
        name = self.data_type
        if 'lfp' in self.data_class:
            name += f"_{self.data_opts['brain_region']}_{self.current_frequency_band}"
        return name

    def merge_dfs(self):
        df_items = list(self.dfs.items())
        df_items.sort(key=lambda item: len(item[1].columns), reverse=True)
        df_names, dfs = zip(*df_items)
        result = dfs[0].copy()
        original_dtypes = result.dtypes

        for df in dfs[1:]:
            common_columns = list(set(result.columns).intersection(set(df.columns)))
            print("Common columns before handling non-identical values:", common_columns)

            if 'experiment' in df.columns:
                df = df.drop(columns=['experiment'])
            for col in common_columns:
                if col in df.columns and col in result.columns:
                    # Ensure data types are the same before merging
                    if result[col].dtype != df[col].dtype:
                        df[col] = df[col].astype(result[col].dtype)
                    if not result[col].equals(df[col]):
                        df = df.rename(columns={col: f'{col}_non_identical'})
                        print(f"Renamed {col} to {col}_non_identical due to non-identical values")

            common_columns = list(set(result.columns).intersection(set(df.columns)))
            print("Common columns after handling non-identical values:", common_columns)

            if common_columns:
                result = pd.merge(result, df, how='left', suffixes=('', '_right'))
            else:
                print("No common columns found for merging. Skipping this DataFrame.")

        # Convert columns back to original data types
        for col, dtype in original_dtypes.items():
            if col in result.columns:
                result[col] = result[col].astype(dtype)

        new_df_name = '_'.join(df_names)
        self.dfs[new_df_name] = result
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

        Notes:
        - The function sets default attributes to include like 'period_type' and then adds to them based on the
          object's `data_class` and `data_type` attributes.
        - Inclusion criteria are set based on the data class, data type, and other options set in the object's
          `data_opts` attribute.
        - The `self.get_data` method is then called with the determined level, inclusion criteria, and attributes to
          collect the rows of data.
        """

        other_attributes = ['block_type']
        inclusion_criteria = []
        if 'lfp' in self.data_class:
            if self.data_type in ['mrl']:
                level = 'mrl_calculator'
                other_attributes += ['frequency', 'fb', 'neuron_type']  # TODO: figure out what fb should be changed to post refactor
                inclusion_criteria.append(lambda x: x.is_valid)
            else:
                level = self.data_opts['row_type']
                if self.data_opts['time_type'] == 'continuous' and self.data_opts.get('power_deviation'):
                    other_attributes.append('power_deviation')
        elif self.data_class == 'behavior':
            level = self.data_opts['row_type']
        else:
            other_attributes += ['category', 'neuron_type']
            level = self.data_opts['row_type']

        if level in ['block', 'event', 'mrl_calculator']:
            inclusion_criteria += [lambda x: find_ancestor_attribute(x, 'block_type') in self.data_opts.get(
                'block_types', ['tone'])]
        if self.data_opts.get('selected_animals') is not None:
            inclusion_criteria += [lambda x: find_ancestor_id(x, 'animal') in self.data_opts['selected_animals']]

        return self.get_data(level, inclusion_criteria, other_attributes)

    def get_data(self, level, inclusion_criteria, other_attributes):
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

        sources = getattr(experiment, f'all_{level}s')
        if self.data_opts.get('frequency_type') == 'continuous':
            sources = [frequency_bin for source in sources for frequency_bin in source.frequency_bins]
        if self.data_opts.get('time_type') == 'continuous':
            sources = [time_bin for source in sources for time_bin in source.time_bins]
        sources = [source for source in sources if all([criterion(source) for criterion in inclusion_criteria])]

        for source in sources:
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

    def make_spreadsheet(self, df_name=None, path=None, force_recalc=True, name_suffix=None):
        """
        Creates a spreadsheet (CSV file) from a specified DataFrame stored within the object.

        This function constructs the filename based on various attributes and options set in the objects `data_opts`. If
        the file already exists and `force_recalc` is set to False, the function will not overwrite the existing file.

        Parameters:
        - df_name (str, optional): Name of the DataFrame to be written to the spreadsheet. Defaults to the object's
          `data_type` attribute if not provided.
        - path (str, optional): Directory path where the spreadsheet should be saved. If not provided, it defaults to
          the `data_path` attribute from the object's `data_opts`.
        - force_recalc (bool, optional): If set to True, the function will overwrite an existing file with the same
          name. Defaults to True.
        - name_suffix (str, optional): String to append to a spreadsheet name to allow writing multiple similar files to
          the same directory.  Defaults to None, which means a random six character string will be written to the file.

        Returns:
        None: The function saves the spreadsheet to the specified path and updates the `spreadsheet_fname` attribute
           of the object.
        """
        if not len(self.dfs):
            self.make_df()
        df_name = df_name if df_name else self.data_type
        if not path:
            path = self.data_opts.get('data_path')
            if 'mrl' in df_name or 'power' in df_name:
                path = os.path.join(path, 'lfp', self.data_type)
            else:
                path = os.path.join(path, self.data_type)
        if not os.path.exists(path):
            os.mkdir(path)
        name_suffix = name_suffix if name_suffix is not None else self.name_suffix
        self.spreadsheet_fname = os.path.join(path, df_name + '_' + name_suffix + '.csv')
        if os.path.exists(self.spreadsheet_fname) and not force_recalc:
            return
        try:
            with open(self.spreadsheet_fname, 'w', newline='') as f:
                self.write_csv(f, df_name)
        except OSError:
            self.spreadsheet_fname = self.spreadsheet_fname[0:75] + self.name_suffix
            if name_suffix != self.name_suffix:
                self.spreadsheet_fname += name_suffix
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
        error_suffix = '/unit' if self.data_opts['row_type'] == 'event' else ''
        error_suffix = error_suffix + '/time_bin' if self.data_opts['post_hoc_bin_size'] > 1 else error_suffix

        if self.data_opts['post_hoc_type'] == 'beta':
            model_formula = f'glmmTMB(formula = {self.data_col} ~ group + (1|animal{error_suffix}), ' \
                            f'family = beta_family(link = "logit"), data = data)'
            interaction_model_formula = f'glmmTMB(formula = {self.data_col} ~ group * neuron_type + ' \
                                        f'(1|animal{error_suffix}), ' \
                                        f'family = beta_family(link = "logit"), data = sub_df)'
            p_val_index = 'coefficients$cond[2, 4]'
            interaction_p_val_index = 'coefficients$cond["groupstressed:neuron_typePN", 4]'
            zero_adjustment_line = f'df$"{self.data_col}"[df$"{self.data_col}" == 0] ' \
                                   f'<- df$"{self.data_col}"[df$"{self.data_col}" == 0] + 1e-6'
        elif self.data_opts['post_hoc_type'] == 'lmer':
            model_formula = f'lmer({self.data_col} ~ group +  (1|animal{error_suffix}), data = data)'
            interaction_model_formula = f'lmer({self.data_col} ~ group * neuron_type + (1|animal{error_suffix}), ' \
                                        f'data = sub_df)'
            p_val_index = 'coefficients[2, 5]'
            interaction_p_val_index = 'coefficients[4, 5]'
            zero_adjustment_line = ''

        return fr'''
                library(glmmTMB)
                library(lme4)
                library(lmerTest)  # for lmer p-values
                library(readr)

                df <- read.csv('{self.spreadsheet_fname}', comment.char="#")

                # Convert variables to factors
                factor_vars <- c('unit', 'animal', 'neuron_type', 'group')
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
                for (time_bin in unique(df$time_bin)) {{
                    # Subset the data for the current time bin
                    sub_df <- df[df$time_bin == time_bin,]

                    # Perform the interaction analysis
                    interaction_model <- {interaction_model_formula}
                    p_val_interact <- summary(interaction_model)${interaction_p_val_index}

                    # Perform the within-category analyses
                    p_val_PN <- perform_regression(sub_df[sub_df$neuron_type == 'PN', ], 'PN')
                    p_val_IN <- perform_regression(sub_df[sub_df$neuron_type == 'IN', ], 'IN')

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

    def get_post_hoc_results(self, force_recalc=True):
        post_hoc_path = os.path.join(self.data_opts['data_path'], self.data_type, 'post_hoc')
        self.make_spreadsheet(path=post_hoc_path)
        self.results_path = os.path.join(post_hoc_path, 'r_post_hoc_results.csv')
        if not os.path.exists(self.results_path) or force_recalc:
            getattr(self, f"write_post_hoc_r_script")()
            subprocess.run(['Rscript', self.script_path], check=True)
        results = pd.read_csv(self.results_path)
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

            factor_vars <- c('animal', 'group', 'block_type', 'neuron_type', 'unit')
            df[factor_vars] <- lapply(df[factor_vars], factor)

            mrl_column <- sym(paste0(frequency_band, "_mrl"))

            averaged_over_unit_result <- df %>%
                group_by(animal, period_type, block, neuron_type, group, frequency_bin) %>%
                summarize(
                    mean_mrl = mean(!!mrl_column, na.rm = TRUE)
                ) %>%
                ungroup()

            data_to_return <- averaged_over_unit_result

            if (evoked) {{
                pretone_data <- subset(averaged_over_unit_result, block_type == "pretone")
                tone_data <- subset(averaged_over_unit_result, block_type == "tone")

                merged_data <- merge(pretone_data, tone_data, by = setdiff(names(averaged_over_unit_result), c("mean_mrl", "block_type")))
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

            groups <- c('control', 'stressed')
            neuron_types <- c('IN', 'PN')

            if (!evoked) {{

                # mrl ~ period_type + period + (1|animal) within each combination of control/stressed and IN/PN
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

                # evoked_mrl ~ neuron_type + period + (1/animal) within control/stressed
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




