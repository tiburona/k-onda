import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.signal import medfilt
import numpy as np
import sys
import os

# Add the main folder to sys.path
main_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_folder_path)

from plotters import Plotter, PeriStimulusPlotter, GroupStatsPlotter
from phy_interface import PhyInterface
from initialize_experiment import Initializer
from misc_data_init.opts_library import FIGURE_1_OPTS, PSTH_OPTS, PROPORTION_OPTS, GROUP_STAT_PSTH_OPTS, GROUP_STAT_PROPORTION_OPTS

plt.rcParams['font.family'] = 'Arial'

init = Initializer('/Users/katie/likhtik/IG_INED_Safety_Recall/init_config.json')
init.init_experiment()
expt = init.experiment


class Figure(Plotter):
    """A class to create a three row figure for the safety paper."""
    def __init__(self, experiment, graph_opts=None, plot_type='standalone'):
        super().__init__(experiment, graph_opts=graph_opts, plot_type=plot_type)
        self.initialize(PSTH_OPTS, FIGURE_1_OPTS)
        self.fig = plt.figure()
        self.grid = GridSpec(3, 1, height_ratios=[3, 4, 4])
        self.rows = []
        self.scatterplot = None

    def spike_data_figure(self):
        self.grid.update(hspace=0.4)
        self.spike_data_figure_first_row()
        self.results_row(PSTH_OPTS, GROUP_STAT_PSTH_OPTS, 1)
        self.results_row(PROPORTION_OPTS, GROUP_STAT_PROPORTION_OPTS, 2)
        plt.show()

    def spike_data_figure_first_row(self):
        self.rows.append(GridSpecFromSubplotSpec(1, 3, subplot_spec=self.grid[0], width_ratios=[3, 3, 4], wspace=0.3))
        self.phy_graphs()
        self.get_subplot_ax(self.rows[0][2], invisible=True)
        self.pn_in_scatterplot()

    def phy_graphs(self):
        ax1, ax2 = (self.get_subplot_ax(self.rows[0][i]) for i in range(2))
        keys = ['data_path', 'animal_id', 'unit_ids', 'electrodes_for_feature', 
                'el_inds', 'pc_inds', 'neuron_type_colors', 'annot_coords']
        (data_path, animal_id, unit_ids, electrodes_for_feature, el_inds, pc_inds, colors, 
         annot_coords) = (self.graph_opts[key] for key in keys)
        phy_interface = PhyInterface(data_path, animal_id)
        for i, unit_id in enumerate(unit_ids):
            animal = [anml for anml in self.experiment.all_animals if anml.identifier == animal_id][0]
            unit = animal.get_child_by_identifier(str(unit_id))
            cluster_id = unit.cluster_id
            waveform = medfilt(unit.waveform, kernel_size=5)
            x, y = phy_interface.one_feature_view(cluster_id, electrodes_for_feature, el_inds, pc_inds)
            ax1.scatter(x, y, alpha=0.3, color=colors[unit.neuron_type], s=5)
            ax2.plot(np.arange(len(waveform)), waveform, color=colors[unit.neuron_type])
            if i == 1:  # We're only putting the FWHM markers on the second line
                min_y, max_y = np.min(waveform), np.max(waveform)
                max_x = np.argmax(waveform)  # find index of max
                half_min = (min_y + max_y) / 2
                # Draw lines indicating amplitude
                ax2.hlines([min_y, max_y], xmin=[max_x - 2, max_x - 2], xmax=[max_x + 2, max_x + 2], color='.2', lw=.7)
                ax2.vlines(max_x, ymin=min_y, ymax=max_y, color='.2', lw=.7)
                # Find indices where waveform is equal to half_min
                half_min_indices = np.where(np.isclose(waveform, half_min, rtol=3e-2))
                # Draw line connecting points at FWHM
                if half_min_indices[0].size > 0:
                    fwhm_start = half_min_indices[0][0]
                    fwhm_end = half_min_indices[0][-1]
                    ax2.hlines(half_min, xmin=fwhm_start, xmax=fwhm_end, color='.2', lw=.7)
                    ax2.text(fwhm_start - 5, half_min, 'FWHM', fontsize=7, ha='right')
        # TODO fix this label
        self.label_phy_graph(ax1, 'Electrode 10, PC 1', 'Electrode 10, PC 2', '(a)', annot_coords)
        self.label_phy_graph(ax2, 'Samples (30k Hz)', '\u03BCV', '(b)', annot_coords)

    def label_phy_graph(self, ax, xlabel, ylabel, letter, annot_coords):
        ax.set_xlabel(xlabel, fontsize=7)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.annotate(letter, xy=annot_coords, xycoords='axes fraction')
        ax.tick_params(axis='both', which='major', labelsize=5, length=1.25)

    def pn_in_scatterplot(self):

        gs0 = GridSpec(100, 100)  # scatterplot needs custom position to take advantage of whitespace below
        overlay_ax = self.get_subplot_ax(gs0[0:30, 65:95], invisible=True)
        overlay_ax.annotate('(c)', self.graph_opts['annot_coords'], xycoords="axes fraction")

        self.scatterplot = GridSpecFromSubplotSpec(3, 3, subplot_spec=overlay_ax.get_subplotspec(), wspace=1, hspace=1)
        ax1 = self.get_subplot_ax(self.scatterplot[:2, 1:])
        units = self.experiment.all_units
        if self.data_opts.get('neuron_quality'):
            units = [unit for unit in units if unit.quality in self.data_opts['neuron_quality']]
        x = [unit.fwhm_microseconds for unit in units]
        y = [unit.firing_rate for unit in units]
        colors = [self.graph_opts['neuron_type_colors'][unit.neuron_type] for unit in units]
        ax1.scatter(x, y, color=colors, alpha=0.5)
        ax1.set_xlabel('FWHM (\u03BCs)', fontsize=7)
        ax1.set_ylabel('Firing Rate (Hz)', fontsize=7)

        ax2 = self.make_neuron_hist(x, position=np.s_[:2, 0], rotate=True, lim_to_flip='xlim', ticks_axes=('y', 'x'),
                                    invisible_spines=('left', 'top'), count_axis='x')
        ax3 = self.make_neuron_hist(x, position=np.s_[2, 1:], rotate=False, lim_to_flip='ylim', ticks_axes=('x', 'y'),
                                    invisible_spines=('right', 'bottom'), count_axis='y')

        ax3.xaxis.set_ticks([])
        ax3.xaxis.set_ticklabels([])

        [ax.tick_params(axis='both', which='major', labelsize=5, length=1.25) for ax in [ax1, ax2, ax3]]

    def make_neuron_hist(self, x, position, rotate, lim_to_flip, ticks_axes, invisible_spines, count_axis):
        ax = self.fig.add_subplot(self.scatterplot[position])
        kwargs = {'bins': 30, 'color': self.graph_opts['hist_color'], 'alpha': 0.8}
        if rotate:
            kwargs['orientation'] = 'horizontal'
        ax.hist(x, **kwargs)
        flipped_lim = getattr(ax, f'get_{lim_to_flip}')()[::-1]
        getattr(ax, f'set_{lim_to_flip}')(flipped_lim)
        [ax.spines[spine].set_visible(False) for spine in invisible_spines]
        ax.tick_params(axis=ticks_axes[0], left=False, labelleft=False, right=False, labelright=False)
        ax.tick_params(axis=ticks_axes[1], left=True, labelleft=True, right=False, labelright=False)
        getattr(ax, f"set_{count_axis}label")('Count', fontsize=7)
        return ax

    def results_row(self, data_opts, group_stat_opts, grid_position):
        letters = [('(d)', '(e)'), ('(f)', '(g)')]
        self.rows.append(GridSpecFromSubplotSpec(1, 2, subplot_spec=self.grid[grid_position], width_ratios=[5, 3],
                                                 hspace=.4))
        ps_plotter = self.initialize_plotter(PeriStimulusPlotter, data_opts, FIGURE_1_OPTS, 2, 2, 0,
                                             letters[grid_position-1][0])
        ps_plotter.plot_groups_data()
        stats_plotter = self.initialize_plotter(GroupStatsPlotter, group_stat_opts, FIGURE_1_OPTS, 2, 1, 1,
                                                letters[grid_position-1][1])
        stats_plotter.plot_group_stats_data()

    def initialize_plotter(self, plotter_class, data_opts, graph_opts, rows, cols, position_in_row, letter):
        plotter = plotter_class(self.experiment, graph_opts=graph_opts, plot_type='gridspec_subplot')
        plotter.data_opts = data_opts
        gridspec = GridSpecFromSubplotSpec(rows, cols, subplot_spec=self.rows[-1][position_in_row], hspace=0.9)
        invisible_ax = self.get_subplot_ax(self.rows[-1][position_in_row], invisible=True)
        invisible_ax.annotate(letter, xycoords="axes fraction", xy=self.graph_opts['annot_coords'])
        plotter.set_gridspec_axes(self.fig, gridspec, rows, cols, invisible_ax=invisible_ax)
        return plotter

    def get_subplot_ax(self, gridspec_slice, invisible=False):
        ax1 = plt.Subplot(self.fig, gridspec_slice)
        self.fig.add_subplot(ax1)
        if invisible:
            ax1.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
            for position in ['top', 'right', 'bottom', 'left']:
                ax1.spines[position].set_visible(False)
        return ax1


figure = Figure(expt, FIGURE_1_OPTS)
# cProfile.run('figure.spike_data_figure()', 'output.prof')
figure.spike_data_figure()

