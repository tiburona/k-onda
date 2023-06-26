import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import ttest_rel

from context import Base


class Stats(Base):
    def __init__(self, experiment, data_type_context, data_opts):
        self.experiment = experiment
        self.data_type_context = data_type_context
        self.data_opts = data_opts
        self.df = None
        self.time_type = self.data_opts['time']
        self.time_col = 'time_point' if self.time_type == 'continuous' else 'period'
        self.data_col = 'rate' if self.data_type == 'psth' else 'proportion'

    def make_df(self):
        self.df = pd.DataFrame([row for unit in self.experiment.all_units for row in self.get_row(unit)])

    def get_row(self, unit):
        unit_num = unit.identifier
        animal = unit.animal.identifier
        condition = unit.animal.condition
        category = unit.neuron_type
        row_dict = {'unit_num': unit_num, 'animal': animal, 'condition': condition, 'category': category}
        return getattr(self, f"{self.time_type}_rows")(unit.data, row_dict)

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
        return [{**row_dict, **{'time_point': time_point, self.data_col: data[time_point]}}
                for time_point in range(self.data_opts.get('num_bins'))]

    def get_post_hoc_results(self):
        self.make_df()
        # average over units, because some animals have only one unit per condition
        avg_df = self.df.groupby(['animal', 'condition', 'category', 'time_point'], as_index=False).mean()
        results = [self.run_post_hocs(avg_df, time_bin) for time_bin in range(self.data_opts['num_bins'])]
        return results

    def run_post_hocs(self, avg_df, time_bin):
        time_subset = avg_df.query(f"time_point == {time_bin}")
        model_def = f"{self.data_col} ~ condition * category"
        interaction_model = smf.mixedlm(model_def, data=time_subset, groups=time_subset['animal']).fit()
        interaction_p = interaction_model.pvalues['condition[T.stressed]:category[T.PN]']

        within_condition_post_hocs = {}
        for condition in self.experiment.conditions:
            time_condition_subset = time_subset.query(f"condition == '{condition}'")
            within_condition_model = smf.mixedlm(f"{self.data_col} ~ category", data=time_condition_subset,
                                                 groups=time_condition_subset['animal']).fit()
            within_condition_post_hocs[condition] = within_condition_model.pvalues['category[T.PN]']

        return interaction_p, within_condition_post_hocs








