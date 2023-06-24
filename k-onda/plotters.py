import os
import math
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from math_functions import get_positive_frequencies, get_spectrum_fenceposts
from utils import smart_title_case, formatted_now, dynamic_property


class Plotter:
    def __init__(self, experiment, data_type_context, neuron_type_context, graph_opts=None):
        self.experiment = experiment
        self.data_type_context = data_type_context
        self.neuron_type_context = neuron_type_context
        self.fig = None
        self.axs = None
        self.y_min = float('inf')
        self.y_max = float('-inf')
        self.fname = ''
        self.title = ''
        self.dir_tags = None
        self.graph_opts = graph_opts
        self.neuron_types = ['PN', 'IN']

    neuron_type = dynamic_property('neuron_type',
                                   getter=lambda self: self.neuron_type_context.val,
                                   setter=lambda self, neuron_type: self.neuron_type_context.set_val(neuron_type))
    data_opts = dynamic_property('data_opts',
                                 getter=lambda self: self.data_type_context.val,
                                 setter=lambda self, opts: self.data_type_context.set_val(opts))
    dtype = dynamic_property('dtype', lambda self: self.data_opts.get('data_type'))

    def initialize(self, data_opts, graph_opts, neuron_type):
        self.y_min = float('inf')
        self.y_max = float('-inf')
        self.graph_opts = graph_opts
        self.data_opts = data_opts
        self.neuron_type = neuron_type

    def plot(self, data_opts, graph_opts, level, neuron_type=None):
        self.initialize(data_opts, graph_opts, neuron_type)
        if level == 'group':
            self.plot_groups(self.experiment.groups)
        elif level == 'animal':
            for group in self.experiment.groups:
                self.plot_animals(group)
        elif level == 'unit':
            [self.plot_units(animal) for group in self.experiment.groups for animal in group.children]
        else:
            print('unrecognized plot type')

    def plot_groups(self, groups):
        self.fig, self.axs = plt.subplots(2, 2, figsize=(15, 15))
        self.fig.subplots_adjust(wspace=0.2, hspace=0.2)
        for row, group in enumerate(groups):
            for col, neuron_type in enumerate(self.neuron_types):
                self.neuron_type = neuron_type
                self.make_subplot(group, row, col, title=f"{group.identifier} {neuron_type}")
        self.neuron_type = None
        self.set_y_scales()
        self.close_plot('groups')

    def plot_animals(self, group):
        num_animals = len(group.children)
        nrows = math.ceil(num_animals / 3)
        self.fig, self.axs = plt.subplots(nrows, 3, figsize=(15, nrows * 6))
        self.fig.subplots_adjust(top=0.85, hspace=0.3, wspace=0.4)

        for i in range(nrows * 3):  # iterate over all subplots
            if i < num_animals:
                row = i // 3  # index based on 3 columns
                col = i % 3  # index based on 3 columns
                animal = group.children[i]
                self.make_subplot(animal, row, col, f"{animal.identifier} {animal.selected_neuron_type}")
            else:
                # Get the axes for the extra subplot and make it invisible
                self.axs[i // 3, i % 3].set_visible(False)

        self.set_y_scales()
        self.close_plot(group.identifier)

    def plot_units(self, animal):
        multi = 2 if self.dtype == 'psth' else 1

        for i in range(0, len(animal.children), self.graph_opts['units_in_fig']):
            n_subplots = min(self.graph_opts['units_in_fig'], len(animal.children) - i)
            self.fig = plt.figure(figsize=(15, 3 * multi * n_subplots))
            self.fig.subplots_adjust(bottom=0.14)
            gs = GridSpec(n_subplots * multi, 1, figure=self.fig)

            for j in range(i, i + n_subplots):
                if self.dtype == 'psth':
                    axes = [self.fig.add_subplot(gs[2 * (j - i), 0]), self.fig.add_subplot(gs[2 * (j - i) + 1, 0])]
                elif self.dtype in ['autocorr', 'spectrum']:
                    axes = [self.fig.add_subplot(gs[j - i, 0])]
                self.plot_unit(animal.children[j], axes)

            self.set_units_plot_frame_and_spacing()

            marker2 = min(i + self.graph_opts['units_in_fig'], len(animal.children))
            self.close_plot(f"{animal.identifier} unit {i + 1} to {marker2}")

    def plot_unit(self, unit, axes):
        if self.dtype == 'psth':
            self.add_raster(unit, axes)
        subplotter = Subplotter(unit, self.data_opts, self.graph_opts, axes[-1])
        plotting_func = getattr(subplotter, f"plot_{self.dtype}")
        plotting_func()
        if self.graph_opts.get('sem'):
            subplotter.add_sem()

    def add_raster(self, unit, axes):
        subplotter = Subplotter(unit, self.data_opts, self.graph_opts, axes[0])
        subplotter.y = unit.get_spikes_by_trials()  # overwrites subplotter.y defined by data_type, which is psth
        subplotter.plot_raster()

    def set_units_plot_frame_and_spacing(self):
        # Add a big subplot without frame and set the x and y labels for this subplot
        big_subplot = self.fig.add_subplot(111, frame_on=False)
        big_subplot.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        big_subplot.set_xlabel(self.get_labels('unit')[self.dtype][0], labelpad=30, fontsize=14)
        plt.subplots_adjust(hspace=0.5)  # Add space between subplots

    def make_subplot(self, data_source, row, col, title=''):
        subplotter = Subplotter(data_source, self.data_opts, self.graph_opts, self.axs[row, col])
        subplotter.plot_data()
        if self.graph_opts.get('sem'):
            subplotter.add_sem()
        self.prettify_subplot(row, col, data_source.name, title=title, y_min=min(subplotter.y), y_max=max(subplotter.y))

    def close_plot(self, basename):
        self.set_dir_and_filename(basename)
        if self.graph_opts.get('footer'):
            self.make_footer()
        self.save_and_close_fig()

    def get_labels(self, level):
        adjustment = self.data_opts.get('adjustment')
        Hz = '' if adjustment == 'normalized' else ' Hz'

        return {'psth': ('Time (s)', f'{adjustment.capitalize()} Firing Rate{Hz}'),
                'proportion_score': ('Time (s)', 'Mean Proportion Positive Normalized Rate'),
                'autocorr': ('Lags (s)', 'Autocorrelation'),
                'spectrum': ('Frequencies (Hz)', 'One-Sided Spectrum')}

    def set_labels(self, row, col, level):
        [getattr(self.axs[row, col], f"set_{dim}label")(self.get_labels(level)[self.dtype][i])
         for i, dim in enumerate(['x', 'y'])]

    def get_ylim(self, row, col, y_min, y_max):
        self.y_min = min(self.y_min, self.axs[row, col].get_ylim()[0], y_min)
        self.y_max = max(self.y_max, self.axs[row, col].get_ylim()[1], y_max)

    def set_y_scales(self):
        if self.graph_opts['equal_y_scales']:
            [ax.set_ylim(self.y_min, self.y_max) for ax in self.axs.flatten()]

    def prettify_subplot(self, row, col, level, title, y_min, y_max):
        self.get_ylim(row, col, y_min, y_max)
        self.set_labels(row, col, level)
        self.axs[row, col].set_title(title)

    def make_footer(self):
        text_vals = [('bin size', self.data_opts['bin_size']), ('selected trials', self.join_trials(' ')),
                     ('time generated', formatted_now())]
        [text_vals.append((k, self.data_opts[k])) for k in ['adjustment, average_method'] if k in self.data_opts]
        text = '  '.join([f"{k}: {v}" for k, v in text_vals])
        if self.dtype in ['autocorr', 'spectrum']:
            ac_vals = [('program', self.data_opts['ac_program']),
                       ('method', self.data_opts['ac_key'])]
            ac_text = '  '.join([f"{k}: {v}" for k, v in ac_vals])
            text = f"{text}\n{ac_text}"
        self.fig.text(0.5, 0.02, text, ha='center', va='bottom', fontsize=15)

    def set_dir_and_filename(self, basename):
        tags = [self.dtype]
        self.dir_tags = tags + [f"trials_{self.join_trials('_')}"]
        tags.insert(0, basename)
        if self.neuron_type:
            tags += [self.neuron_type]
        self.title = smart_title_case(' '.join([tag.replace('_', ' ') for tag in tags]))
        self.fig.suptitle(self.title, weight='bold', y=.95, fontsize=20)
        if self.dtype in ['autocorr', 'spectrum']:
            for key in ['ac_program', 'ac_key']:
                tags += [self.data_opts[key]]
        self.fname = f"{'_'.join(tags)}.png"

    def save_and_close_fig(self):
        dirs = [self.graph_opts['graph_dir']]
        if self.dir_tags is not None:
            dirs += self.dir_tags
        path = os.path.join(*dirs)
        os.makedirs(path, exist_ok=True)
        self.fig.savefig(os.path.join(path, self.fname))
        plt.close(self.fig)

    def join_trials(self, s):
        return s.join([str(t) for t in self.data_opts['trials']])


class Subplotter:
    def __init__(self, data_source, data_opts, graph_opts, ax):
        self.data_source = data_source
        self.d_opts = data_opts
        self.dtype = data_opts['data_type']
        self.g_opts = graph_opts
        self.ax = ax
        self.x = None
        self.y = data_source.data

    def set_limits_and_ticks(self, x_min, x_max, x_tick_min, x_step, y_min=None, y_max=None):
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_xticks(np.arange(x_tick_min, x_max, step=x_step))
        if y_min is not None and y_max is not None:
            self.ax.set_ylim(y_min, y_max)

    def set_labels(self, x_label='', y_label='', fontsize=13):
        self.ax.set_xlabel(x_label, fontsize=fontsize)
        self.ax.set_ylabel(y_label, fontsize=fontsize)

    def plot_raster(self):
        opts = self.d_opts
        for i, spiketrain in enumerate(self.y):
            for spike in spiketrain:
                self.ax.vlines(spike, i + .5, i + 1.5)
        self.set_labels(y_label='Trial')
        self.set_limits_and_ticks(-opts['pre_stim'], opts['post_stim'], self.g_opts['tick_step'], .5, len(self.y) + .5)
        self.ax.add_patch(plt.Rectangle((0, self.ax.get_ylim()[0]), 0.05, self.ax.get_ylim()[1] - self.ax.get_ylim()[0],
                                        facecolor='gray', alpha=0.3))

    def plot_bar(self, width, x_min, x_max, num, x_tick_min, x_step, y_min=None, y_max=None, x_label='', y_label='',
                 color='k'):
        self.x = np.linspace(x_min, x_max, num=num)
        self.ax.bar(self.x, self.y, width=width, color=color)
        self.set_limits_and_ticks(x_min, x_max, x_tick_min, x_step, y_min, y_max)
        self.set_labels(x_label=x_label, y_label=y_label)

    def plot_psth(self):
        opts = self.d_opts
        self.plot_bar(width=opts['bin_size'], x_min=-opts['pre_stim'], x_max=opts['post_stim'], num=len(self.y),
                      x_tick_min=0, x_step=self.g_opts['tick_step'], y_label='Relative Spike Rate (Hz)')
        self.ax.fill_betweenx([min(self.y), max(self.y)], 0, 0.05, color='k', alpha=0.2)

    def plot_proportion_score(self):
        self.plot_psth()

    def plot_autocorr(self):
        opts = self.d_opts
        self.plot_bar(width=opts['bin_size']*.95, x_min=opts['bin_size'],  x_max=opts['max_lag']*opts['bin_size'],
                      num=opts['max_lag'], x_tick_min=0, x_step=self.g_opts['tick_step'], y_min=min(self.y) - .01,
                      y_max=max(self.y) + .01)

    def plot_spectrum(self):
        freq_range, max_lag, bin_size = (self.d_opts.get(opt) for opt in ['freq_range', 'max_lag', 'bin_size'])
        first, last = get_spectrum_fenceposts(freq_range, max_lag, bin_size)
        self.x = get_positive_frequencies(max_lag, bin_size)[first:last]
        self.ax.plot(self.x, self.y)

    def plot_data(self):
        getattr(self, f"plot_{self.dtype}")()

    def add_sem(self):
        opts = self.d_opts
        if opts['data_type'] in ['autocorr', 'spectrum'] and opts['ac_key'] == self.data_source.name + '_by_rates':
            print("It doesn't make sense to add standard error to a graph of autocorr over rates.  Skipping.")
            return
        sem = self.data_source.get_sem()
        self.ax.fill_between(self.x, self.y - sem, self.y + sem, color='blue', alpha=0.2)

