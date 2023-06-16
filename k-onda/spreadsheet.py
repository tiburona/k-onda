import numpy as np
import csv


class Spreadsheet:

    def make_spreadsheet(self, opts, units, path, row_type='continuous'):
        # Open the file with write permissions
        with open(path, 'w', newline='') as f:
            # Create a csv writer
            writer = csv.writer(f)

            # Write the header row
            header = ['unit_num', 'animal', 'condition', 'category', 'period', 'rate']
            writer.writerow(header)

            # For each unit, get rows and write them to the csv
            for unit in units:
                rows = self.get_row(unit, opts, row_type)
                for row in rows:
                    writer.writerow(row.values())

    def get_row(self, unit, opts, row_type):
        unit_num = unit.identifier
        animal = unit.parent.identifier
        condition = unit.parent.parent.identifier
        category = unit.neuron_type
        psth = unit.get_psth(opts)
        row_dict = {'unit_num': unit_num, 'animal': animal, 'condition': condition, 'category': category}

        if row_type == 'binned':
            return self.binned_rows(psth, row_dict)
        else:
            return self.continuous_rows(psth, row_dict)

    @staticmethod
    def calculate_rate(psth, bin_slice):
        bins = psth[slice(*bin_slice)]
        return np.mean(bins)

    def binned_rows(self, psth, row_dict):
        periods = ['during_beep', 'early_post_beep', 'mid_post_beep', 'late_post_beep']
        bin_slices = [(0, 5), (5, 30), (30, 60), (60, 100)]
        rows = []
        for period, bin_slice in zip(periods, bin_slices):
            rate = self.calculate_rate(psth, bin_slice)
            rows.append({**row_dict, **{'period': period,'rate': rate}})
        return rows

    def continuous_rows(self, psth, row_dict):
        rows = []
        for time_point in range(65):  # Assuming 65 time points
            rate = psth[time_point]  # Assuming psth is already calculated per time point
            rows.append({**row_dict, **{'time_point': time_point, 'rate': rate}})
        return rows
