import os
import math
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from signal_processing import get_positive_frequencies, get_spectrum_fenceposts
from utils import smart_title_case


class Plotter:
    def __init__(self, opts, data_type=None, equal_y_scales=True):
        self.fig = None
        self.axs = None
        self.y_min = float('inf')
        self.y_max = float('-inf')
        self.fname = ''
        self.title = ''
        self.dir_tags = None
        self.dtype = data_type if data_type is not None else opts['data_type']
        self.opts = opts
        self.equal_y_scales = equal_y_scales
        self.labels = {'psth': ('Time (s)', 'Firing Rate (Hz'), 'autocorr': ('Lags (s)',  'Autocorrelation'),
                       'spectrum': ('Frequencies (Hz)',  'One-Sided Spectrum')}

    def plot_animals(self, group, neuron_type=None, footer=True, sem=False):
        num_animals = len(group.animals)
        nrows = math.ceil(num_animals / 3)
        self.fig, self.axs = plt.subplots(nrows, 3, figsize=(15, nrows * 5))
        self.fig.subplots_adjust(top=0.9)

        for i in range(nrows * 3):  # iterate over all subplots
            row = i // 3  # index based on 3 columns
            col = i % 3  # index based on 3 columns
            if i < num_animals:
                animal = group.animals[i]
                title = f"{animal.identifier} {neuron_type}"
                self.make_subplot(animal, row, col, title=title, neuron_type=neuron_type, sem=sem)
            else:  # if there's no animal for this subplot
                self.axs[row, col].axis('off')  # hide this subplot

        self.prettify_plot()
        if footer:
            self.make_footer()
        self.set_dir_and_filename(group.identifier, neuron_type=neuron_type)
        self.save_and_close_fig()

    def plot_groups(self, groups, neuron_types, footer=True, sem=False):
        self.fig, self.axs = plt.subplots(2, 2, figsize=(15, 15))  # Create a 2x2 subplot grid
        self.fig.subplots_adjust(wspace=0.2, hspace=0.2)  # adjust the spacing between subplots
        for row, group in enumerate(groups):
            for col, neuron_type in enumerate(neuron_types):
                self.make_subplot(group, row, col, title=f"{group.identifier} {neuron_type}", neuron_type=neuron_type,
                                  sem=sem)
        self.set_dir_and_filename('groups', neuron_type=None)
        if footer:
            self.make_footer()
        self.prettify_plot()
        self.save_and_close_fig()

    def plot_units(self, animal, sem=False):
        multi = 2 if self.dtype == 'psth' else 1

        for i in range(0, len(animal.units['good']), self.opts['units_in_fig']):
            n_subplots = min(self.opts['units_in_fig'], len(animal.units['good']) - i)
            self.fig = plt.figure(figsize=(10, 3 * multi * n_subplots))
            gs = GridSpec(n_subplots * multi, 1, figure=self.fig)

            for j in range(i, i + n_subplots):
                if self.dtype == 'psth':
                    axes = [self.fig.add_subplot(gs[2 * (j - i), 0]), self.fig.add_subplot(gs[2 * (j - i) + 1, 0])]
                elif self.dtype in ['autocorr', 'spectrum']:
                    axes = [self.fig.add_subplot(gs[j - i, 0])]
                self.plot_unit(animal.units['good'][j], axes, sem=sem)

            # Add a big subplot without frame and set the x and y labels for this subplot
            big_subplot = self.fig.add_subplot(111, frame_on=False)
            big_subplot.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            big_subplot.set_xlabel(self.labels[self.opts['data_type']][0], labelpad=30)
            big_subplot.set_ylabel(self.labels[self.opts['data_type']][1], labelpad=30)

            plt.subplots_adjust(hspace=0.5)  # Add space between subplots
            marker2 = min(i + self.opts['units_in_fig'], len(animal.units['good']))
            self.fname = f"{animal.identifier}_unit_{i + 1}_to_{marker2}.png"
            self.set_dir_and_filename(f"{animal.identifier}_unit_{i + 1}_to_{marker2}")
            self.save_and_close_fig()

    def plot_unit(self, unit, axes, sem=False):
        if self.dtype == 'psth':
            subplotter = Subplotter(axes[0])
            subplotter.y = unit.get_trials_spikes(self.opts)
            subplotter.plot_raster(self.opts)
        subplotter = Subplotter(axes[-1])
        plotting_func = getattr(subplotter, f"plot_{self.dtype}")
        data_func = getattr(unit, f"get_{self.dtype}")
        subplotter.y = data_func(self.opts)
        plotting_func(self.opts)
        if sem:
            subplotter.add_sem(unit, self.opts)

    def make_subplot(self, data_source, row, col, title='', neuron_type=None, sem=False):
        subplotter = Subplotter(self.axs[row, col])
        subplotter.y = data_source.get_data(self.dtype)
        if np.all(np.isnan(subplotter.y)):
            self.axs[row, col].axis('off')  # hide this subplot
        else:
            getattr(subplotter, f"plot_{self.dtype}")(self.opts)
            if sem:
                subplotter.add_sem(data_source, self.opts, neuron_type=neuron_type)
            self.prettify_subplot(row, col, title=title, y_min=min(subplotter.y), y_max=max(subplotter.y))

    def set_labels(self, row, col):
        [getattr(self.axs[row, col], f"set_{dim}label")(self.labels[self.dtype][i]) for i, dim in enumerate(['x', 'y'])]

    def get_ylim(self, row, col, y_min, y_max):
        self.y_min = min(self.y_min, self.axs[row, col].get_ylim()[0], y_min)
        self.y_max = max(self.y_max, self.axs[row, col].get_ylim()[1], y_max)

    def set_y_scales(self):
        if self.equal_y_scales:
            [ax.set_ylim(self.y_min, self.y_max) for ax in self.axs.flatten()]

    def prettify_subplot(self, row, col, title, y_min, y_max):
        self.get_ylim(row, col, y_min, y_max)
        self.set_labels(row, col)
        self.axs[row, col].set_title(title)

    def prettify_plot(self):
        self.set_y_scales()
        plt.subplots_adjust(hspace=0.7)  # Add space between subplots
        # self.fig.tight_layout()

    def make_footer(self):
        now = datetime.now()
        formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
        text_vals = [('bin size', self.opts['bin_size']), ('selected trials', self.join_trials(' ')),
                     ('time generated', formatted_now)]
        text = '  '.join([f"{k}: {v}" for k, v in text_vals])
        if self.dtype in ['autocorr', 'spectrum']:
            ac_vals = [('program', self.opts['ac_program']), ('method', self.opts['ac_key'])]
            ac_text = '  '.join([f"{k}: {v}" for k, v in ac_vals])
            text = f"{text}\n{ac_text}"
        self.fig.text(0.5, 0.02, text, ha='center', va='bottom', fontsize=15)

    def set_dir_and_filename(self, basename, neuron_type=None):
        tags = [self.dtype]
        self.dir_tags = tags + [f"trials_{self.join_trials('_')}"]
        tags.insert(0, basename)
        if neuron_type:
            tags += [neuron_type]
        self.title = smart_title_case(' '.join(tags))
        self.title = self.title.replace('_', ' ')
        self.fig.suptitle(self.title, weight='bold', y=.95, fontsize=20)
        for key in ['ac_program', 'ac_key']:
            tags += [self.opts[key]]
        self.fname = f"{'_'.join(tags)}.png"

    def save_and_close_fig(self):
        dirs = [self.opts['graph_dir']]
        if self.dir_tags is not None:
            dirs += self.dir_tags
        path = os.path.join(*dirs)
        os.makedirs(path, exist_ok=True)
        self.fig.savefig(os.path.join(path, self.fname))
        plt.close(self.fig)

    def join_trials(self, s):
        return s.join([str(t) for t in self.opts['trials']])

    def ac_str(self, s):
        for (old, new) in [('pd', 'Pandas'), ('np', 'NumPy'), ('ml', 'Matlab')]:
            s = s.replace(old, new)


class Subplotter:
    def __init__(self, ax):
        self.ax = ax
        self.x = None
        self.y = None

    def set_limits_and_ticks(self, x_min, x_max, x_tick_min, x_step, y_min=None, y_max=None):
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_xticks(np.arange(x_tick_min, x_max, step=x_step))
        if y_min is not None and y_max is not None:
            self.ax.set_ylim(y_min, y_max)

    def set_labels(self, x_label='', y_label=''):
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)

    def plot_raster(self, opts):
        for i, spiketrain in enumerate(self.y):
            for spike in spiketrain:
                self.ax.vlines(spike, i + .5, i + 1.5)
        self.set_labels(y_label='Trial')
        self.set_limits_and_ticks(-opts['pre_stim'], opts['post_stim'], opts['tick_step'], .5, len(self.y) + .5)
        self.ax.add_patch(plt.Rectangle((0, self.ax.get_ylim()[0]), 0.05, self.ax.get_ylim()[1] - self.ax.get_ylim()[0],
                                        facecolor='gray', alpha=0.3))

    def plot_bar(self, width, x_min, x_max, num, x_tick_min, x_step, y_min=None, y_max=None, x_label='', y_label='',
                 color='k'):
        self.x = np.linspace(x_min, x_max, num=num)
        self.ax.bar(self.x, self.y, width=width, color=color)
        self.set_limits_and_ticks(x_min, x_max, x_tick_min, x_step, y_min, y_max)
        self.set_labels(x_label=x_label, y_label=y_label)

    def plot_psth(self, opts):
        self.plot_bar(width=opts['bin_size'], x_min=-opts['pre_stim'], x_max=opts['post_stim'], num=len(self.y),
                      x_tick_min=0, x_step=opts['tick_step'], y_label='Relative Spike Rate (Hz)')
        self.ax.fill_betweenx([min(self.y), max(self.y)], 0, 0.05, color='k', alpha=0.2)

    def plot_autocorr(self, opts):
        self.plot_bar(width=opts['bin_size']*.95, x_min=opts['bin_size'],  x_max=opts['max_lag']*opts['bin_size'],
                      num=opts['max_lag'], x_tick_min=0, x_step=opts['tick_step'], y_min=0, y_max=max(self.y) + .01)

    def plot_spectrum(self, opts):
        first, last = get_spectrum_fenceposts(opts)
        self.x = get_positive_frequencies(opts['max_lag'], opts['bin_size'])[first:last]
        self.ax.plot(self.x, self.y)

    def add_sem(self, data_source, opts, neuron_type=None):
        if opts['data_type'] in ['autocorr', 'spectrum'] and opts['ac_key'] == data_source.name + '_by_rates':
            print("It doesn't make sense to add standard error to this kind of graph.  Skipping.")
            return
        sem = data_source.get_sem(opts, neuron_type=neuron_type)
        self.ax.fill_between(self.x, self.y - sem, self.y + sem, color='blue', alpha=0.2)
