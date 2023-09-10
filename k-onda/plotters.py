import os
import math
import numpy as np
import seaborn as sns
import pandas as pd
from itertools import product

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
# Create a custom legend
from matplotlib.patches import Patch

from math_functions import get_positive_frequencies, get_spectrum_fenceposts, std_err
from plotting_helpers import smart_title_case, formatted_now, PlottingMixin
from data import Base
from stats import Stats


class Plotter(Base):
    """Makes plots, where a plot is a display of particular kind of data.  For displays of multiple plots of multiple
    kinds of data, see the figure module."""

    def __init__(self, experiment, data_type_context, neuron_type_context, period_type_context, graph_opts=None,
                 lfp=None, plot_type='standalone'):
        self.experiment = experiment
        self.data_type_context = data_type_context
        self.data_opts = None
        self.neuron_type_context = neuron_type_context
        self.selected_neuron_type = None
        self.period_type_context = period_type_context
        self.selected_period_type = None
        self.lfp = lfp
        self.graph_opts = graph_opts
        self.plot_type = plot_type
        self.fig = None
        self.axs = None
        self.y_min = float('inf')
        self.y_max = float('-inf')
        self.fname = ''
        self.title = ''
        self.dir_tags = None
        self.stats = None
        self.full_axes = None
        self.invisible_ax = None
        self.grid = None

    def initialize(self, data_opts, graph_opts, neuron_type=None):
        """Both initializes values on self and sets values for the contexts and all the contexts' subscribers."""
        self.graph_opts = graph_opts
        self.data_opts = data_opts  # Sets data_opts for all subscribers to data_type_context
        self.selected_neuron_type = neuron_type  # Sets neuron type for all subscribers to neuron_type_context

    def close_plot(self, basename):
        self.set_dir_and_filename(basename)
        if self.graph_opts.get('footer'):
            self.make_footer()
        self.save_and_close_fig()

    def save_and_close_fig(self):
        dirs = [self.graph_opts['graph_dir']]
        if self.dir_tags is not None:
            dirs += self.dir_tags
        path = os.path.join(*dirs)
        os.makedirs(path, exist_ok=True)
        self.fig.savefig(os.path.join(path, self.fname))
        plt.close(self.fig)


class PeriStimulusPlotter(Plotter, PlottingMixin):
    """Makes plots where the x-axis is time around the stimulus, and y can be a variety of types of data."""

    def __init__(self, experiment, data_type_context, neuron_type_context, graph_opts=None, plot_type='standalone'):
        super().__init__(experiment, data_type_context, neuron_type_context, graph_opts=graph_opts, plot_type=plot_type)
        self.multiplier = 1 if self.plot_type == 'standalone' else 0.5

    def plot(self, data_opts, graph_opts, level=None, neuron_type=None):
        self.initialize(data_opts, graph_opts, neuron_type)
        if level == 'group':
            self.plot_groups()
        elif level == 'animal':
            for group in self.experiment.groups:
                self.plot_animals(group)
        elif level == 'unit':
             [self.plot_units(animal) for group in self.experiment.groups for animal in group.children]
        else:
            print('unrecognized plot type')

    def plot_groups(self):
        self.fig, self.axs = plt.subplots(2, 2, figsize=(15, 15))
        self.fig.subplots_adjust(wspace=0.2, hspace=0.2)
        self.plot_groups_data()
        self.close_plot('groups')

    def plot_groups_data(self):
        for row, neuron_type in enumerate(self.neuron_types):
            self.selected_neuron_type = neuron_type
            for col, group in enumerate(self.experiment.groups):
                self.make_subplot(group, self.axs[row, col], title=f"{group.identifier.capitalize()} {neuron_type}")
        self.selected_neuron_type = None
        self.set_y_scales()
        self.set_pip_patches()
        self.set_labels()

    def plot_animals(self, group):
        num_animals = len(group.children)
        nrows = math.ceil(num_animals / 3)
        self.fig, self.axs = plt.subplots(nrows, 3, figsize=(15, nrows * 6))
        self.fig.subplots_adjust(top=0.85, hspace=0.3, wspace=0.4)

        for i in range(nrows * 3):  # iterate over all subplots
            if i < num_animals:
                row = i // 3  # index based on 3 columns
                col = i % 3  # index based on 3 columns
                ax = self.axs[row, col] if nrows > 1 else self.axs[i//3]
                animal = group.children[i]
                self.make_subplot(animal, ax, f"{animal.identifier} {animal.selected_neuron_type}")
            else:
                # Get the axes for the extra subplot and make it invisible
                ax_ind = (i // 3, i % 3) if nrows > 1 else (i // 3,)
                self.axs[ax_ind].set_visible(False)

        self.set_y_scales()
        self.set_pip_patches()
        self.close_plot(group.identifier)

    def make_subplot(self, data_source, ax, title=''):
        subplotter = PeriStimulusSubplotter(data_source, self.data_opts, self.graph_opts, ax,
                                            self.plot_type, multiplier=self.multiplier)
        subplotter.plot_data()
        if self.graph_opts.get('sem'):
            subplotter.add_sem()
        self.prettify_subplot(ax, title=title, y_min=min(subplotter.y), y_max=max(subplotter.y))

    def plot_units(self, animal):
        multi = 2 if self.data_type == 'psth' else 1
        for i in range(0, len(animal.children), self.graph_opts['units_in_fig']):
            n_subplots = min(self.graph_opts['units_in_fig'], len(animal.children) - i)
            self.fig = plt.figure(figsize=(15, 3 * multi * n_subplots))
            self.fig.subplots_adjust(bottom=0.14)
            gs = GridSpec(n_subplots * multi, 1, figure=self.fig)

            for j in range(i, i + n_subplots):
                if self.data_type == 'psth':
                    axes = [self.fig.add_subplot(gs[2 * (j - i), 0]), self.fig.add_subplot(gs[2 * (j - i) + 1, 0])]
                elif self.data_type in ['autocorr', 'spectrum', 'proportion']:
                    axes = [self.fig.add_subplot(gs[j - i, 0])]
                self.plot_unit(animal.children[j], axes)

            self.set_units_plot_frame_and_spacing()

            marker2 = min(i + self.graph_opts['units_in_fig'], len(animal.children))
            self.close_plot(f"{animal.identifier} unit {i + 1} to {marker2}")

    def plot_unit(self, unit, axes):
        if self.data_type == 'psth':
            self.add_raster(unit, axes)
        subplotter = PeriStimulusSubplotter(unit, self.data_opts, self.graph_opts, axes[-1])
        plotting_func = getattr(subplotter, f"plot_{self.data_type}")
        plotting_func()
        if self.graph_opts.get('sem'):
            subplotter.add_sem()

    def add_raster(self, unit, axes):
        subplotter = PeriStimulusSubplotter(unit, self.data_opts, self.graph_opts, axes[0])
        subplotter.y = unit.get_spikes_by_trials()  # overwrites subplotter.y defined by data_type, which is psth
        subplotter.plot_raster()

    def set_units_plot_frame_and_spacing(self):
        # Add a big subplot without frame and set the x and y labels for this subplot
        big_subplot = self.fig.add_subplot(111, frame_on=False)
        big_subplot.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        big_subplot.set_xlabel(self.get_labels()[self.data_type][0], labelpad=30, fontsize=14)
        plt.subplots_adjust(hspace=0.5)  # Add space between subplots

    def get_ylim(self, ax, y_min, y_max):
        self.y_min = min(self.y_min, ax.get_ylim()[0], y_min)
        self.y_max = max(self.y_max, ax.get_ylim()[1], y_max)

    def set_y_scales(self):
        if self.graph_opts['equal_y_scales']:
            [ax.set_ylim(self.y_min, self.y_max) for ax in self.axs.flatten()]

    def set_pip_patches(self):
        [ax.fill_betweenx([self.y_min, self.y_max], 0, 0.05, color='k', alpha=0.2) for ax in self.axs.flatten()]

    def prettify_subplot(self, ax, title, y_min, y_max):
        self.get_ylim(ax, y_min, y_max)
        ax.set_title(title, fontsize=17*self.multiplier)

    def make_footer(self):
        text_vals = [('bin size', self.data_opts['bin_size']), ('selected trials', self.join_trials(' ')),
                     ('time generated', formatted_now())]
        [text_vals.append((k, self.data_opts[k])) for k in ['adjustment, average_method'] if k in self.data_opts]
        text = '  '.join([f"{k}: {v}" for k, v in text_vals])
        if self.data_type in ['autocorr', 'spectrum']:
            ac_vals = [('program', self.data_opts['ac_program']),
                       ('method', self.data_opts['ac_key'])]
            ac_text = '  '.join([f"{k}: {v}" for k, v in ac_vals])
            text = f"{text}\n{ac_text}"
        self.fig.text(0.5, 0.02, text, ha='center', va='bottom', fontsize=15)

    def set_dir_and_filename(self, basename):
        tags = [self.data_type]
        self.dir_tags = tags + [f"trials_{self.join_trials('_')}"]
        tags.insert(0, basename)
        if self.selected_neuron_type:
            tags += [self.selected_neuron_type]
        self.title = smart_title_case(' '.join([tag.replace('_', ' ') for tag in tags]))
        self.fig.suptitle(self.title, weight='bold', y=.95, fontsize=20)
        if self.data_type in ['autocorr', 'spectrum']:
            for key in ['ac_program', 'ac_key']:
                tags += [self.data_opts[key]]
        if self.data_opts.get('base'):
            tags += [self.data_opts.get('base')]
        self.fname = f"{'_'.join(tags)}.png"


    def join_trials(self, s):
        return s.join([str(t) for t in self.data_opts['trials']])

    def set_gridspec_axes(self, fig, gridspec, numrows, numcols, invisible_ax=None):
        self.fig = fig
        self.grid = gridspec

        if numrows > 1 and numcols > 1:
            self.axs = np.array([
                [plt.Subplot(fig, gridspec[row, col]) for col in range(numcols)]
                for row in range(numrows)
            ])
        elif numrows == 1 or numcols == 1:  # For case where only one row or one column
            self.axs = np.array([plt.Subplot(fig, gridspec[i]) for i in range(max(numrows, numcols))])
        else:
            raise ValueError("Number of rows or columns must be greater than zero.")

        for ax in np.ravel(self.axs):
            fig.add_subplot(ax)

        self.invisible_ax = invisible_ax


class PeriStimulusSubplotter(PlottingMixin):
    """Constructs a subplot of a PeriStimulusPlot."""
    def __init__(self, data_source, data_opts, graph_opts, ax, parent_type, multiplier=1):
        self.data_source = data_source
        self.data_opts = data_opts
        self.data_type = data_opts['data_type']
        self.g_opts = graph_opts
        self.ax = ax
        self.x = None
        self.y = data_source.data
        self.parent_type = parent_type
        self.multiplier = multiplier
        self.plot_type = 'subplot'

    def set_limits_and_ticks(self, x_min, x_max, x_tick_min, x_step, y_min=None, y_max=None):
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_xticks(np.arange(x_tick_min, x_max, step=x_step))
        self.ax.tick_params(axis='both', which='major', labelsize=10*self.multiplier, length=5*self.multiplier,
                            width=2*self.multiplier)

        if y_min is not None and y_max is not None:
            self.ax.set_ylim(y_min, y_max)

    def plot_raster(self):
        opts = self.data_opts
        for i, spiketrain in enumerate(self.y):
            for spike in spiketrain:
                self.ax.vlines(spike, i + .5, i + 1.5)
        self.set_labels(y_label='Trial')
        self.set_limits_and_ticks(-opts['pre_stim'], opts['post_stim'], self.g_opts['tick_step'], .5, len(self.y) + .5)
        self.ax.add_patch(plt.Rectangle((0, self.ax.get_ylim()[0]), 0.05, self.ax.get_ylim()[1] - self.ax.get_ylim()[0],
                                        facecolor='gray', alpha=0.3))

    def plot_bar(self, width, x_min, x_max, num, x_tick_min, x_step, y_min=None, y_max=None, x_label='', y_label='',
                 color='k', facecolor='white'):
        if self.data_source.name == 'group' and 'group_colors' in self.g_opts:
            color = self.g_opts['group_colors'][self.data_source.identifier]
        else:
            color = 'black'
        self.x = np.linspace(x_min, x_max, num=num)
        self.ax.bar(self.x, self.y, width=width, color=color)
        self.ax.set_facecolor(facecolor)
        self.ax.patch.set_alpha(0.2)
        self.set_limits_and_ticks(x_min, x_max, x_tick_min, x_step, y_min, y_max)
        if self.parent_type == 'standalone':
            self.set_labels(x_and_y_labels=(x_label, y_label))

    def plot_psth(self):
        opts = self.data_opts
        xlabel, ylabel = self.get_labels()[self.data_opts['data_type']]
        self.plot_bar(width=opts['bin_size'], x_min=-opts['pre_stim'], x_max=opts['post_stim'], num=len(self.y),
                      x_tick_min=0, x_step=self.g_opts['tick_step'], y_label=ylabel)

    def plot_proportion(self):
        self.plot_psth()

    def plot_autocorr(self):
        opts = self.data_opts
        self.plot_bar(width=opts['bin_size']*.95, x_min=opts['bin_size'],  x_max=opts['max_lag']*opts['bin_size'],
                      num=opts['max_lag'], x_tick_min=0, x_step=self.g_opts['tick_step'], y_min=min(self.y) - .01,
                      y_max=max(self.y) + .01)

    def plot_spectrum(self):
        freq_range, max_lag, bin_size = (self.data_opts.get(opt) for opt in ['freq_range', 'max_lag', 'bin_size'])
        first, last = get_spectrum_fenceposts(freq_range, max_lag, bin_size)
        self.x = get_positive_frequencies(max_lag, bin_size)[first:last]
        self.ax.plot(self.x, self.y)

    def plot_data(self):
        getattr(self, f"plot_{self.data_type}")()

    def add_sem(self):
        opts = self.data_opts
        if opts['data_type'] in ['autocorr', 'spectrum'] and opts['ac_key'] == self.data_source.name + '_by_rates':
            print("It doesn't make sense to add standard error to a graph of autocorr over rates.  Skipping.")
            return
        sem = self.data_source.get_sem()
        self.ax.fill_between(self.x, self.y - sem, self.y + sem, color='blue', alpha=0.2)


class GroupStatsPlotter(PeriStimulusPlotter):

    def __init__(self, experiment, data_type_context, neuron_type_context, graph_opts=None, plot_type='standalone'):
        super().__init__(experiment, data_type_context, neuron_type_context, graph_opts=graph_opts, plot_type=plot_type)

    def plot_group_stats(self):
        self.fig, self.axs = plt.subplots(2, 1, figsize=(15, 15))
        self.current_ax = None
        self.plot_group_stats_data()
        self.close_plot('stats_plot')

    def plot_group_stats_data(self):
        self.stats = Stats(self.experiment, self.data_type_context, self.neuron_type_context, self.data_opts)
        force_recalc = self.graph_opts['force_recalc'] # TODO: this isn't doing anything.
        interaction_ps, neuron_type_specific_ps = self.stats.get_post_hoc_results()

        bin_size = self.data_opts.get('bin_size')
        for row, neuron_type in enumerate(self.neuron_types):
            self.selected_neuron_type = neuron_type
            for group in self.experiment.groups:
                color = self.graph_opts['group_colors'][group.identifier]
                x = np.arange(len(group.data)) * bin_size
                y = group.data
                if max(y) > self.y_max:
                    self.y_max = max(y)
                if min(y) < self.y_min:
                    self.y_min = min(y)
                self.axs[row].plot(x, y, label=group.identifier, color=color)
                sem = group.get_sem()
                self.axs[row].fill_between(x, y - sem, y + sem, color=color, alpha=0.2)

            self.axs[row].set_title(f"{neuron_type}", fontsize=17*self.multiplier, loc='left')
            self.axs[row].set_xticks(np.arange(self.data_opts['pre_stim'], self.data_opts['post_stim'],
                                                   step=self.graph_opts['tick_step']))
            self.axs[row].tick_params(axis='both', which='major', labelsize=10*self.multiplier,
                                      length=5*self.multiplier, width=2*self.multiplier)
            self.current_ax = self.axs[row]
            self.set_labels()

            # Annotate significant points within conditions
            self.add_significance_markers(neuron_type_specific_ps[neuron_type], 'within_condition', row=row,
                                          y=self.y_max * 1.05)
        for row in range(len(self.neuron_types)):
            self.axs[row].set_ylim(self.y_min * 1.1, self.y_max * 1.1)
            self.axs[row].set_xlim(self.data_opts['pre_stim'], self.data_opts['post_stim'])
            [self.axs[row].spines[side].set_visible(False) for side in ['top', 'right']]

        self.selected_neuron_type = None
        self.add_significance_markers(interaction_ps, 'interaction')
        self.place_legend()

    def add_significance_markers(self, p_values, p_type, row=None, y=None):
        bin_size = self.data_opts.get('bin_size')
        for time_bin, p_value in enumerate(p_values):
            if p_value < .05:
                self.get_significance_markers(row, p_type, time_bin, bin_size, y, p_values)

    def get_significance_markers(self, row, p_type, time_bin, bin_size, y, p_values):
        if p_type == 'within_condition':
            self.axs[row].annotate('*', (time_bin * bin_size, y * 0.85), fontsize=20 * self.multiplier,
                                   ha='center', color='black')
        else:
            self.get_interaction_text(time_bin, p_values)

    def get_interaction_text(self, time_bin, p_values):
        # calculate the x position as a fraction of the plot width
        x = time_bin / len(p_values)
        y = 0.485

        if self.plot_type == 'standalone':
            left = self.axs[0].get_position().xmin
            right = self.axs[0].get_position().xmax
            x = left + (right-left) * x

        else:
            gridspec_position = self.grid.get_subplot_params(self.fig)
            x = gridspec_position.left + (gridspec_position.right - gridspec_position.left) * x
            y = gridspec_position.bottom + (gridspec_position.top - gridspec_position.bottom) / 2 - .01

        self.fig.text(x, y, "\u2020", fontsize=15 * self.multiplier, ha='center', color='black')

    def place_legend(self):
        if self.plot_type == 'standalone':
            x, y = 1, 1
            lines = [mlines.Line2D([], [], color=color, label=condition, linewidth=2 * self.multiplier)
                     for color, condition in zip(['green', 'orange'], ['Control', 'Stressed'])]
            self.fig.legend(handles=lines, loc='upper left', bbox_to_anchor=(x, y), prop={'size': 14 * self.multiplier})


class PiePlotter(Plotter):
    """Constructs a pie chart of up- or down-regulation of individual neurons"""

    def __init__(self, experiment, data_type_context, neuron_type_context, graph_opts=None, plot_type='standalone'):
        super().__init__(experiment, data_type_context, neuron_type_context, graph_opts=graph_opts, plot_type=plot_type)

    def plot_unit_pie_chart(self, data_opts, graph_opts):
        self.initialize(data_opts, graph_opts, neuron_type=None)
        labels = ['Up', 'Down', 'No Change']
        colors = ['red', 'yellow', 'orange']

        for nt in [None, 'PN', 'IN']:
            self.selected_neuron_type = nt
            for group in self.experiment.groups:
                if nt:
                    units = [unit for unit in self.experiment.all_units
                             if (unit.neuron_type == nt and unit.animal.condition == group.identifier)]
                else:
                    units = self.experiment.all_units
                sizes = [len([unit for unit in units if unit.upregulated_to_pip() == num]) for num in [1, -1, 0]]

                self.fig = plt.figure()
                plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
                plt.axis('equal')
                self.close_plot(f'{group.identifier}_during_pip')


class MRLPlotter(Plotter):
    def __init__(self, experiment, data_type_context, neuron_type_context, period_type_context, lfp=None, graph_opts=None,
                 plot_type='standalone'):
        print('hello?')
        super().__init__(experiment, data_type_context, neuron_type_context, period_type_context, lfp=lfp, graph_opts=graph_opts,
                         plot_type=plot_type)

    def rose_plot(self, data_opts, graph_opts):
        self.initialize(data_opts, graph_opts, neuron_type=None)
        self.fig = plt.figure(figsize=(15, 15))

        ncols = 2 if self.data_opts.get('adjustment') == 'relative' else 4

        self.axs = [
            [self.fig.add_subplot(2, ncols, 1 + ncols * row + col, projection='polar') for col in range(ncols)]
            for row in range(2)
        ]

        for i, neuron_type in enumerate(self.neuron_types):
            self.selected_neuron_type = neuron_type
            for j, group in enumerate(self.lfp.groups):
                if self.data_opts.get('adjustment') == 'relative':
                    self.selected_period_type = 'tone'
                    self.make_rose_plot(group, self.axs[i][j], title=f"{group.identifier.capitalize()} {neuron_type}")
                else:
                    for k, period_type in enumerate(['pretone', 'tone']):
                        self.selected_period_type = period_type
                        self.make_rose_plot(group, self.axs[i][j*2 + k],
                                            title=f"{group.identifier.capitalize()} {neuron_type} {period_type}")
        self.selected_neuron_type = None
        self.close_plot('rose_plot')

    def set_dir_and_filename(self, basename):
        tags = [self.lfp.current_frequency_band, self.lfp.brain_region]
        self.dir_tags = [self.data_type]
        for opt in ['phase', 'adjustment']:
            if self.data_opts.get(opt):
                tags += [self.data_opts.get(opt)]
        self.title = smart_title_case(' '.join([tag.replace('_', ' ') for tag in tags]))
        self.fig.suptitle(self.title, weight='bold', y=.95, fontsize=20)
        self.fname = f"{basename}_{'_'.join(tags)}.png"

    def make_rose_plot(self, group, ax, title=""):
        n_bins = 36
        bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
        color = self.graph_opts['group_colors'][group.identifier]
        width = 2 * np.pi / n_bins
        ax.bar(bin_edges[:-1], group.get_angle_counts(), width=width, align='edge', color=color, alpha=1)
        current_ticks = ax.get_yticks()
        ax.set_yticks(current_ticks[::2])  # every second tick
        ax.set_title(title)

    def mrl_vals_plot(self, data_opts, graph_opts):
        self.initialize(data_opts, graph_opts, neuron_type=None)

        data = []
        for neuron_type in self.neuron_types:
            self.selected_neuron_type = neuron_type
            for group in self.lfp.groups:
                for period_type in ['pretone', 'tone']:
                    self.selected_period_type = period_type
                    data.append([neuron_type, group.identifier, period_type, group.data, std_err(group.scatter),
                                 group.scatter])

        df = pd.DataFrame(data, columns=['Neuron Type', 'Group', 'Period', 'Average MRL', 'sem', 'scatter'])

        group_order = df['Group'].unique()
        period_order = ['pretone', 'tone']

        # Define custom color and hatch palettes
        group_colors = self.graph_opts['group_colors']
        period_hatches = {'pretone': '/', 'tone': None}

        g = sns.catplot(data=df, x='Group', y='Average MRL', hue='Period', row='Neuron Type', kind='bar',
                        height=4, aspect=1.5, dodge=True, legend=False, hue_order=period_order, order=group_order)
        g.set_axis_labels("", "Average MRL")
        g.fig.subplots_adjust(top=0.85, hspace=0.4, right=0.85)
        g.despine(left=True)

        # Adjust the appearance
        legend_elements = [Patch(facecolor='white', hatch=period_hatches[stage], edgecolor='black', label=stage.upper())
                           for stage in period_order]

        for ax, neuron_type in zip(g.axes.flat, self.neuron_types):
            bars = ax.patches
            for grp in group_order:
                for prd in period_order:
                    row = df[(df['Neuron Type'] == neuron_type) & (df['Group'] == grp) & (df['Period'] == prd)].iloc[0]

                    # Find corresponding bar
                    idx = group_order.tolist().index(grp) + period_order.index(prd) * len(group_order)
                    bar = bars[idx]

                    # Set color and hatch
                    bar.set_facecolor(group_colors[grp])
                    bar.set_hatch(period_hatches[prd])

                    # Plotting the error bars and scatter points
                    bar_x = bar.get_x() + bar.get_width() / 2
                    ax.errorbar(bar_x, row['Average MRL'], yerr=row['sem'], color='black', capsize=5)

                    jitter = np.random.rand(len(row['scatter'])) * 0.1 - 0.05
                    ax.scatter([bar_x + j for j in jitter], row['scatter'], color='black', s=20)

                ax.axhline(0, color='gray', linestyle='--')
                ax.set_title(ax.get_title(), fontsize=14)

        g.fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, .9))
        self.fig = g.fig
        self.close_plot('average_mrl')
















