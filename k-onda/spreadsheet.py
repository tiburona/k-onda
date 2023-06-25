import numpy as np
import csv
import os

from context import Base


class Spreadsheet(Base):
    def __init__(self, experiment, data_type_context, opts):
        self.experiment = experiment
        self.data_type_context = data_type_context
        self.data_opts = opts

    def make_spreadsheet(self):
        path = self.data_opts.get('path')
        fname = os.path.join(path, '_'.join([self.data_type, self.data_opts['time']]) + '.csv')

        with open(fname, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['unit_num', 'animal', 'condition', 'category', 'period', 'rate']
            writer.writerow(header)

            for unit in self.experiment.all_units:
                rows = self.get_row(unit)
                for row in rows:
                    writer.writerow(row.values())

    def get_row(self, unit):
        time = self.data_opts.get('time')
        unit_num = unit.identifier
        animal = unit.animal.identifier
        condition = unit.animal.condition
        category = unit.neuron_type
        row_dict = {'unit_num': unit_num, 'animal': animal, 'condition': condition, 'category': category}

        if time == 'binned':
            return self.binned_rows(unit.data, row_dict)
        else:
            return self.continuous_rows(unit.data, row_dict)

    @staticmethod
    def calculate_rate(psth, bin_slice):
        bins = psth[slice(*bin_slice)]
        return np.mean(bins)

    def binned_rows(self, data, row_dict):
        periods = ['during_beep', 'early_post_beep', 'mid_post_beep', 'late_post_beep']
        bin_slices = [(0, 5), (5, 30), (30, 60), (60, 100)]
        rows = []
        for period, bin_slice in zip(periods, bin_slices):
            rate = self.calculate_rate(data, bin_slice)
            rows.append({**row_dict, **{'period': period, 'rate': rate}})
        return rows

    @staticmethod
    def continuous_rows(psth, row_dict):
        rows = []
        for time_point in range(65):  # Assuming 65 time points
            rate = psth[time_point]
            rows.append({**row_dict, **{'time_point': time_point, 'rate': rate}})
        return rows

