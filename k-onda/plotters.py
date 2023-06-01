import math

from matplotlib import patches
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from signal_processing import get_positive_frequencies
from utils import smart_title_case


class Plotter:
    def __init__(self, opts, data_type=None, equal_y_scales=True):
        self.fig = None
        self.axs = None
        self.y_min = float('inf')
        self.y_max = float('-inf')
        self.fname = ''
        self.dtype = data_type if data_type is not None else opts['data_type']
        self.opts = opts
        self.equal_y_scales = equal_y_scales
        self.labels = {'psth': ('Time (s)', 'Firing Rate (Hz'), 'autocorr': ('Lags (s)',  'Autocorrelation'),
                       'spectrum': ('Frequencies (Hz)',  'One-Sided Spectrum')}

    def plot_animals(self, group, neuron_type=None):
        num_animals = len(group.animals)
        nrows = math.ceil(num_animals / 3)
        self.fig, self.axs = plt.subplots(nrows, 3, figsize=(15, nrows * 5))
        self.fig.subplots_adjust(top=0.9)

        for i in range(nrows * 3):  # iterate over all subplots
            row = i // 3  # index based on 3 columns
            col = i % 3  # index based on 3 columns
            if i < num_animals:
                animal = group.animals[i]
                subplotter = Subplotter(self.axs[row, col])
                avg = animal.get_average(self.dtype, self.opts, neuron_type=neuron_type)
                if np.all(np.isnan(avg)):
                    self.axs[row, col].axis('off')  # hide this subplot
                else:
                    getattr(subplotter, f"plot_{self.dtype}")(avg, self.opts)
                    self.prettify_subplot(row, col, title=f"{animal.name}")
            else:  # if there's no animal for this subplot
                self.axs[row, col].axis('off')  # hide this subplot

        self.prettify_plot()
        neuron_str = '' if neuron_type is None else f"{neuron_type}_"
        self.fname = f"{group.name}_animals_{neuron_str}{self.dtype}.png"
        self.save_and_close_fig(subdirs=[f"{self.dtype}_{'_'.join([str(t) for t in self.opts['trials']])}"])

    def plot_groups(self, groups, neuron_types):
        self.fig, self.axs = plt.subplots(2, 2, figsize=(15, 15))  # Create a 2x2 subplot grid

        for row, group in enumerate(groups):
            for col, neuron_type in enumerate(neuron_types):
                subplotter = Subplotter(self.axs[row, col])
                avg = group.get_average(self.dtype, self.opts, neuron_type=neuron_type)
                if np.all(np.isnan(avg)):
                    self.axs[row, col].axis('off')  # hide this subplot
                else:
                    getattr(subplotter, f"plot_{self.dtype}")(avg, self.opts)
                    self.prettify_subplot(row, col, title=f"{group.name} {neuron_type} {self.dtype}")

        self.prettify_plot()
        self.fname = f"Groups_{self.dtype}.png"
        self.save_and_close_fig(subdirs=[f"{self.dtype}_{'_'.join([str(t) for t in self.opts['trials']])}"])

    def plot_units(self, animal):
        opts = self.opts
        multi = 2 if self.dtype == 'psth' else 1

        for i in range(0, len(animal.units['good']), opts['units_in_fig']):
            n_subplots = min(opts['units_in_fig'], len(animal.units['good']) - i)
            self.fig = plt.figure(figsize=(10, 3 * multi * n_subplots))
            gs = GridSpec(n_subplots * multi, 1, figure=self.fig)

            for j in range(i, i + n_subplots):
                if self.dtype == 'psth':
                    axes = [self.fig.add_subplot(gs[2 * (j - i), 0]), self.fig.add_subplot(gs[2 * (j - i) + 1, 0])]
                elif self.dtype in ['autocorr', 'spectrum']:
                    axes = [self.fig.add_subplot(gs[j - i, 0])]
                self.plot_unit(animal.units['good'][j], axes)

            # Add a big subplot without frame and set the x and y labels for this subplot
            big_subplot = self.fig.add_subplot(111, frame_on=False)
            big_subplot.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            big_subplot.set_xlabel(self.labels[opts['data_type']][0], labelpad=30)
            big_subplot.set_ylabel(self.labels[opts['data_type']][1], labelpad=30)

            plt.subplots_adjust(hspace=0.5)  # Add space between subplots
            self.fname = f"{animal.name}_unit_{i + 1}_to_{min(i + opts['units_in_fig'], len(animal.units['good']))}.png"
            self.save_and_close_fig(subdirs=[f"{self.dtype}_{'_'.join([str(t) for t in opts['trials']])}"])

    def plot_unit(self, unit, axes):
        if self.opts['data_type'] == 'psth':
            Subplotter(axes[0]).plot_raster(unit.get_trials_spikes(self.opts), self.opts)
        subplotter = Subplotter(axes[-1])
        getattr(subplotter, f"plot_{self.dtype}")(getattr(unit, f"get_{self.dtype}")(self.opts), self.opts)

    def set_labels(self, row, col):
        [getattr(self.axs[row, col], f"set_{dim}label")(self.labels[self.dtype][i]) for i, dim in enumerate(['x', 'y'])]

    def get_ylim(self, row, col):
        self.y_min = min(self.y_min, self.axs[row, col].get_ylim()[0])
        self.y_max = max(self.y_max, self.axs[row, col].get_ylim()[1])

    def set_y_scales(self):
        if self.equal_y_scales:
            [ax.set_ylim(self.y_min, self.y_max) for ax in self.axs.flatten()]

    def prettify_subplot(self, row, col, title):
        self.get_ylim(row, col)
        self.set_labels(row, col)
        self.axs[row, col].set_title(title)

    def prettify_plot(self):
        self.set_y_scales()
        # plt.subplots_adjust(hspace=0.5)  # Add space between subplots
        # self.fig.tight_layout()

    def save_and_close_fig(self, subdirs=None):
        self.fig.suptitle(smart_title_case(self.fname[:-4]), weight='bold', y=1)
        dirs = [self.opts['graph_dir']]
        if subdirs is not None:
            dirs += subdirs
        path = os.path.join(*dirs)
        os.makedirs(path, exist_ok=True)
        self.fig.savefig(os.path.join(path, self.fname))
        plt.close(self.fig)


class Subplotter:
    def __init__(self, ax):
        self.ax = ax

    def set_limits_and_ticks(self, x_min, x_max, x_tick_min, x_step, y_min=None, y_max=None):
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_xticks(np.arange(x_tick_min, x_max, step=x_step))
        if y_min is not None and y_max is not None:
            self.ax.set_ylim(y_min, y_max)

    def set_labels_and_titles(self, x_label='', y_label='', title=''):
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.ax.set_title(title)

    def plot_raster(self, data, opts):
        for i, spiketrain in enumerate(data):
            for spike in spiketrain:
                self.ax.vlines(spike, i + .5, i + 1.5)
        self.set_labels_and_titles(y_label='Trial')
        self.set_limits_and_ticks(-opts['pre_stim'], opts['post_stim'], opts['tick_step'], .5, len(data) + .5)
        self.ax.add_patch(plt.Rectangle((0, self.ax.get_ylim()[0]), 0.05, self.ax.get_ylim()[1] - self.ax.get_ylim()[0],
                                        facecolor='gray', alpha=0.3))

    def plot_bar(self, data, width, x_min, x_max, num, x_tick_min, x_step, y_min=None, y_max=None, x_label='', y_label='',
                 color='k', title=''):
        x = np.linspace(x_min, x_max, num=num)
        self.ax.bar(x, data, width=width, color=color)
        self.set_limits_and_ticks(x_min, x_max, x_tick_min, x_step, y_min, y_max)
        self.set_labels_and_titles(x_label=x_label, y_label=y_label, title=title)

    def plot_psth(self, data, opts):
        self.plot_bar(data, width=opts['bin_size'], x_min=-opts['pre_stim'], x_max=opts['post_stim'],
                      num=len(data), x_tick_min=0, x_step=opts['tick_step'], y_label='Relative Spike Rate (Hz)')
        self.ax.fill_betweenx([min(data), max(data)], 0, 0.05, color='k', alpha=0.2)

    def plot_autocorr(self, data, opts):
        self.plot_bar(data, width=opts['bin_size']*.95, x_min=opts['bin_size'],
                      x_max=opts['lags']*opts['bin_size'], num=opts['lags'], x_tick_min=0,
                      x_step=opts['tick_step'], y_min=0, y_max=max(data) + .05)

    # TODO: fix "up to Hz" code
    def plot_spectrum(self, data, opts):
        # last_index: resolution of the positive spectrum = lags/2 (the number of points in the spectrum)/
        # sampling rate/2 (the range of frequencies in the spectrum).  If up_to_Hz is greater than the highest frequency
        # available, this won't do anything.
        last_index = int(opts['up_to_hz'] * opts['lags'] * opts['bin_size'] + 1)
        x = get_positive_frequencies(opts['lags'], opts['bin_size'])[:last_index]
        y = data[:last_index]
        self.ax.plot(x, y)