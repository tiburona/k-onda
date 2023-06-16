import numpy as np
import csv


class Spreadsheet:

    def make_spreadsheet(self, opts, units, path):
        # Open the file with write permissions
        with open(path, 'w', newline='') as f:
            # Create a csv writer
            writer = csv.writer(f)

            # Write the header row
            header = ['unit_num', 'animal', 'condition', 'category', 'period', 'rate']
            writer.writerow(header)

            # For each unit, get rows and write them to the csv
            for unit in units:
                rows = self.get_row(unit, opts)
                for row in rows:
                    writer.writerow(row.values())

    def get_row(self, unit, opts):
        unit_num = unit.identifier
        animal = unit.parent.identifier
        condition = unit.parent.parent.identifier
        category = unit.neuron_type
        psth = unit.get_psth(opts)
        periods = ['during_beep', 'early_post_beep', 'mid_post_beep', 'late_post_beep']
        bin_slices = [(0, 5), (5, 30), (30, 60), (60, 100)]

        rows = []
        for period, bin_slice in zip(periods, bin_slices):
            rate = self.calculate_rate(psth, bin_slice)
            rows.append({
                'unit_num': unit_num,
                'animal': animal,
                'condition': condition,
                'category': category,
                'period': period,
                'rate': rate
            })
        return rows

    @staticmethod
    def calculate_rate(psth, bin_slice):
        bins = psth[slice(*bin_slice)]
        return np.mean(bins)
