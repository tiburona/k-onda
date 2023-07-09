import re
from datetime import datetime


class PlottingMixin:
    def get_labels(self):
        adjustment = self.data_opts.get('adjustment')
        Hz = '' if adjustment == 'normalized' else ' Hz'

        return {'psth': ('Time (s)', f'{adjustment.capitalize()} Firing Rate{Hz}'),
                'proportion_score': ('Time (s)',''
                                     f'Proportion Positive {self.data_opts.get("base").capitalize() + "s"}'),
                'autocorr': ('Lags (s)', 'Autocorrelation'),
                'spectrum': ('Frequencies (Hz)', 'One-Sided Spectrum')}

    def set_labels(self, x_and_y_labels=(None, None)):
        canonical_labels = self.get_labels()[self.data_type]
        labels = [canonical_labels[i] if label is None else label for i, label in enumerate(x_and_y_labels)]

        if self.plot_type == 'standalone':
            # Create an invisible subplot that spans the entire figure
            big_subplot = self.fig.add_subplot(111, frame_on=False)
            big_subplot.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            object_to_label = big_subplot
            x_coord_of_y_label = -0.08
            object_to_label.yaxis.set_label_coords(x_coord_of_y_label, 0.5)  # For the y-axis

        elif self.plot_type == 'subplot':
            object_to_label = self.ax
        else:
            object_to_label = self.invisible_ax
            if self.__class__.__name__ == 'PeriStimulusPlotter':
                x_coord_of_y_label = -0.08
            else:
                x_coord_of_y_label = -0.15

            object_to_label.xaxis.set_label_coords(0.5, -0.1)  # For the x-axis
            object_to_label.yaxis.set_label_coords(x_coord_of_y_label, 0.5)  # For the y-axis

        [getattr(object_to_label, f"set_{dim}label")(labels[i], fontsize=15 * self.multiplier)
         for i, dim in enumerate(['x', 'y'])]


# def iterate_and_track(func):
#     def wrapper(self, *args, **kwargs):
#         # initialize y_min and y_max
#         self.y_min = float('inf')
#         self.y_max = -float('inf')
#
#         # Iterate over neuron types and groups
#         for row, neuron_type in enumerate(self.neuron_types):
#             self.selected_neuron_type = neuron_type
#             for col, group in enumerate(self.experiment.groups):
#                 color = 'orange' if group.identifier == 'stressed' else 'green'
#                 y = group.data
#
#                 # Call the provided method
#                 if 'row' in func.__code__.co_varnames and 'col' in func.__code__.co_varnames:
#                     result = func(self, row, col, *args, **kwargs)
#                 elif 'row' in func.__code__.co_varnames:
#                     result = func(self, row, *args, **kwargs)
#                 elif 'col' in func.__code__.co_varnames:
#                     result = func(self, col, *args, **kwargs)
#                 else:
#                     result = func(self, *args, **kwargs)
#
#                 # Update y_min and y_max
#                 if result is not None:
#                     self.y_min = min(self.y_min, min(result))
#                     self.y_max = max(self.y_max, max(result))
#
#                 # Rest of the plotting logic remains the same
#                 self.axs[row].plot(x, y, label=group.identifier, color=color)
#                 self.axs[row].set_title(f"{neuron_type}", fontsize=17)
#                 self.axs[row].set_xticks(np.arange(self.data_opts['pre_stim'], self.data_opts['post_stim'],
#                                                    step=self.graph_opts['tick_step']))
#
#         return self.y_min, self.y_max
#
#     return wrapper



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
