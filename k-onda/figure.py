import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np

from plotters import Plotter, PeriStimulusPlotter, GroupStatsPlotter
from plotting_helpers import annotate_subplot
from phy_interface import PhyInterface
from initialize_experiment import experiment as expt, data_type_context as dt_context, neuron_type_context as nt_context
from opts_library import FIGURE_1_OPTS, PSTH_OPTS, PROPORTION_OPTS, GROUP_STAT_PSTH_OPTS, GROUP_STAT_PROPORTION_OPTS

plt.rcParams['font.family'] = 'Arial'


class Figure(Plotter):
    def __init__(self, experiment, data_type_context, neuron_type_context, graph_opts=None, plot_type='standalone'):
        super().__init__(experiment, data_type_context, neuron_type_context, graph_opts=graph_opts, plot_type=plot_type)
        self.fig = plt.figure()
        self.grid = GridSpec(3, 1, height_ratios=[3, 4, 4])
        self.rows = []
        self.annot_coords = (-0.11, 1.1)

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
        keys = ['data_path', 'animal_id', 'cluster_ids', 'electrodes_for_feature', 'electrodes_for_waveform',
                'el_inds', 'pc_inds', 'neuron_type_colors']
        data_path, animal_id, cluster_ids, electrodes_for_feature, electrodes_for_waveform, el_inds, pc_inds, colors = (
            self.graph_opts[key] for key in keys)
        colors = list(colors.values())
        phy_interface = PhyInterface(data_path, animal_id)
        for i, cluster_id in enumerate(cluster_ids):
            x, y = phy_interface.one_feature_view(cluster_id, electrodes_for_feature, el_inds, pc_inds)
            ax1.scatter(x, y, alpha=0.3, color=colors[i], s=5)
            waveform = phy_interface.get_mean_waveforms(cluster_id, electrodes_for_waveform[i])
            ax2.plot(np.arange(len(waveform)), waveform, color=colors[i])
            if i == 1:  # Assume the second line is red (indexed at 1)
                min_y, max_y = np.min(waveform), np.max(waveform)
                max_x = np.argmax(waveform)  # find indices of min and max
                half_min = (min_y + max_y) / 2

                # Draw short horizontal lines
                ax2.hlines([min_y, max_y], xmin=[max_x - 2, max_x - 2], xmax=[max_x + 2, max_x + 2], color='.2', lw=.7)

                # Draw vertical line connecting min and max
                ax2.vlines(max_x, ymin=min_y, ymax=max_y, color='.2', lw=.7)

                # Find indices where waveform is equal to half_min
                half_min_indices = np.where(np.isclose(waveform, half_min, rtol=3e-2))  # adjust tolerance as needed

                # Draw line connecting points at FWHM
                if half_min_indices[0].size > 0:
                    # Identify the starting and ending x-coordinates for the FWHM
                    fwhm_start = half_min_indices[0][0]
                    fwhm_end = half_min_indices[0][-1]

                    # Draw a horizontal line representing the FWHM
                    ax2.hlines(half_min, xmin=fwhm_start, xmax=fwhm_end, color='.2', lw=.7)

                    # Position FWHM label to the left of the FWHM line
                    ax2.text(fwhm_start - 5, half_min, 'FWHM', fontsize=8, ha='right')

        ax1.set_xlabel('Electrode 10, PC 1', fontsize=8)
        ax1.set_ylabel('Electrode 10, PC 2', fontsize=8)
        ax2.set_xlabel('Samples (30k Hz)', fontsize=8)
        ax2.set_ylabel('\u03BCV', fontsize=8)
        annotate_subplot(ax1, '(a)', xy=self.annot_coords, xycoords="axes fraction")
        annotate_subplot(ax2, '(b)', xy=self.annot_coords, xycoords="axes fraction")

    def pn_in_scatterplot(self):
        gs0 = GridSpec(100, 100)
        # Use the slice from row 7 to 10 and column 7 to 10 for overlay_ax
        overlay_ax = self.get_subplot_ax(gs0[0:30, 65:95], invisible=True)
        annotate_subplot(overlay_ax, '(c)', self.annot_coords, xycoords="axes fraction")

        # Create a 3x3 GridSpec within the SubplotSpec of the custom positioned axes
        scatterplot = GridSpecFromSubplotSpec(3, 3, subplot_spec=overlay_ax.get_subplotspec(), wspace=1, hspace=1)
        ax1 = self.get_subplot_ax(scatterplot[:2, 1:])
        x = [unit.fwhm_microseconds for unit in self.experiment.all_units]
        y = [unit.firing_rate for unit in self.experiment.all_units]
        colors = [self.graph_opts['neuron_type_colors'][unit.neuron_type] for unit in self.experiment.all_units]
        ax1.scatter(x, y, color=colors, alpha=0.5)
        ax1.set_xlabel('FWHM (\u03BCs)', fontsize=7)
        ax1.set_ylabel('Firing Rate (Hz)', fontsize=7)

        ax2 = self.fig.add_subplot(scatterplot[:2, 0])
        ax2.hist(y, bins=30, orientation='horizontal', color='#9678D3', alpha=.8)
        ax2.set_xlim(ax2.get_xlim()[::-1])
        ax2.tick_params(axis='y', left=False, labelleft=False, right=False, labelright=False)
        ax2.tick_params(axis='x', bottom=True, labelbottom=True, right=False, labelright=False)
        ax2.set_xlabel('Count', fontsize=7)
        ax2.spines['left'].set_visible(False)
        ax2.spines['top'].set_visible(False)

        ax3 = self.get_subplot_ax(scatterplot[2, 1:])
        ax3.hist(x, bins=30, color='#9678D3', alpha=.8)
        ax3.set_ylim(ax3.get_ylim()[::-1])
        ax3.spines['right'].set_visible(False)
        ax3.spines['bottom'].set_visible(False)
        ax3.tick_params(axis='x', left=False, labelleft=False, right=False, labelright=False)
        ax3.xaxis.set_ticks([])
        ax3.xaxis.set_ticklabels([])
        ax3.tick_params(axis='y', left=True, labelleft=True, right=False, labelright=False)

        [ax.tick_params(axis='both', which='major', labelsize=5, length=1.25) for ax in [ax1, ax2, ax3]]

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
        plotter = plotter_class(self.experiment, self.data_type_context, self.neuron_type_context,
                                graph_opts=graph_opts, plot_type='gridspec_subplot')
        plotter.data_opts = data_opts
        gridspec = GridSpecFromSubplotSpec(rows, cols, subplot_spec=self.rows[-1][position_in_row], hspace=0.9)
        invisible_ax = self.get_subplot_ax(self.rows[-1][position_in_row], invisible=True)
        invisible_ax.annotate(letter, xycoords="axes fraction", xy=self.annot_coords)
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

figure = Figure(expt, dt_context, nt_context, FIGURE_1_OPTS)
figure.spike_data_figure()

