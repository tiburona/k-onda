import re
from datetime import datetime


class PlottingMixin:
    def get_labels(self):
        adjustment = self.data_opts.get('adjustment')
        Hz = '' if adjustment == 'normalized' else ' Hz'

        return {'psth': ('Time (s)', f'{adjustment.capitalize()} Firing Rate{Hz}'),
                'proportion_score': ('Time (s)',
                                     f'Proportion Positive {self.data_opts.get("base").capitalize() + "s"}'),
                'autocorr': ('Lags (s)', 'Autocorrelation'),
                'spectrum': ('Frequencies (Hz)', 'One-Sided Spectrum')}

    def set_labels(self, x_and_y_labels=(None, None)):
        canonical_labels = self.get_labels()[self.data_type]
        labels = [canonical_labels[i] if label is None else label for i, label in enumerate(x_and_y_labels)]
        if self.plot_type == 'standalone':
            object_to_label = self.fig
        elif self.plot_type == 'subplot':
            object_to_label = self.ax
        else:
            object_to_label = self.invisible_ax
            object_to_label.xaxis.set_label_coords(0.5, -0.2)  # For the x-axis
            object_to_label.yaxis.set_label_coords(-.2, 0.5)  # For the y-axis

        [getattr(object_to_label, f"set_{dim}label")(labels[i])
         for i, dim in enumerate(['x', 'y'])]


def formatted_now():
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


def smart_title_case(s):
    lowercase_words = {'a', 'an', 'the', 'at', 'by', 'for', 'in', 'of', 'on', 'to', 'up', 'and', 'as', 'but', 'or',
                       'nor', 'is'}
    acronyms = {'psth'}
    words = re.split(r'(\W+)', s)  # Split string on non-alphanumeric characters, preserving delimiters
    title_words = []
    for i, word in enumerate(words):
        if word.lower() in lowercase_words and i != 0 and i != len(words) - 1:
            title_words.append(word.lower())
        elif word.lower() in acronyms:
            title_words.append(word.upper())
        elif not word.isupper():
            title_words.append(word.capitalize())
        else:
            title_words.append(word)
    title = ''.join(title_words)
    return title


def ac_str(s):
    for (old, new) in [('pd', 'Pandas'), ('np', 'NumPy'), ('ml', 'Matlab')]:
        s = s.replace(old, new)
