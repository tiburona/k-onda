import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np

from plotters import Plotter, PeriStimulusPlotter, GroupStatsPlotter
from phy_interface import PhyInterface
from initialize_experiment import experiment as expt, data_type_context as dt_context, neuron_type_context as nt_context
from opts_library import GROUP_STAT_OPTS, PROPORTION_OPTS, FIGURE_1_OPTS

plt.rcParams['font.family'] = 'Arial'


class Figure(Plotter):
    def __init__(self, experiment, data_type_context, neuron_type_context, graph_opts=None, plot_type='standalone'):
        super().__init__(experiment, data_type_context, neuron_type_context, graph_opts=graph_opts, plot_type=plot_type)
        self.fig = plt.figure()
        self.grid = GridSpec(2, 1, height_ratios=[3, 4])
        self.rows = []

    def spike_data_figure(self):
        self.grid.update(hspace=0.4)
        self.spike_data_figure_first_row()
        self.spike_data_figure_second_row()
        plt.show()

    def spike_data_figure_first_row(self):

        self.rows.append(GridSpecFromSubplotSpec(1, 3, subplot_spec=self.grid[0], width_ratios=[3, 3, 4], wspace=0.3))
        self.phy_graphs()
        self.pn_in_scatterplot()

    def phy_graphs(self):
        ax1, ax2 = (self.get_subplot_ax(self.rows[0][i]) for i in range(2))
        keys = ['data_path', 'animal_id', 'cluster_ids', 'electrodes', 'el_inds', 'pc_inds', 'neuron_type_colors']
        data_path, animal_id, cluster_ids, electrodes, el_inds, pc_inds, colors = (self.graph_opts[key] for key in keys)
        colors = list(colors.values())
        phy_interface = PhyInterface(data_path, animal_id)
        for i, cluster_id in enumerate(cluster_ids):
            x, y = phy_interface.one_feature_view(cluster_id, electrodes, el_inds, pc_inds)
            ax1.scatter(x, y, alpha=0.3, color=colors[i], s=5)
            waveform = phy_interface.get_mean_waveforms(cluster_id, electrodes)
            ax2.plot(np.arange(len(waveform)), waveform, color=colors[i])
        ax2.set_xlabel('Samples (30k Hz)', fontsize=8)
        ax2.set_ylabel('\u03BCV', fontsize=8)

        for ax in [ax1, ax2]:
            ax.tick_params(axis='both', which='major', labelsize=5, length=1.25)

    def pn_in_scatterplot(self):
        scatterplot = GridSpecFromSubplotSpec(3, 3, subplot_spec=self.rows[0][2], wspace=0.2)

        # Large subplot in latter two columns and first two rows
        ax1 = self.get_subplot_ax(scatterplot[:2, 1:])
        x = [unit.fwhm_microseconds for unit in self.experiment.all_units]
        y = [unit.firing_rate for unit in self.experiment.all_units]
        colors = [self.graph_opts['neuron_type_colors'][unit.neuron_type] for unit in self.experiment.all_units]
        ax1.scatter(x, y, color=colors, alpha=0.5)  # Scatter plot on ax1
        ax1.xaxis.set_ticks([])
        ax1.yaxis.set_ticks([])

        ax2 = self.fig.add_subplot(scatterplot[:2, 0])
        ax2.hist(y, bins=30, orientation='horizontal', color='purple', alpha=0.5)
        ax2.set_xlim(ax2.get_xlim()[::-1])  # Flip the x-axis

        ax2.tick_params(axis='y', left=True, labelleft=True, right=False, labelright=False)
        ax2.set_ylabel('Firing Rate (Hz)', fontsize=7)
        ax2.tick_params(axis='x', bottom=True, labelbottom=True, right=False, labelright=False)
        ax2.set_xlabel('Count', fontsize=7)

        # Small subplot in the last row and latter two columns
        ax3 = self.get_subplot_ax(scatterplot[2, 1:])
        ax3.hist(x, bins=30, color='purple', alpha=0.5)  # Histogram on ax2
        ax3.set_xlabel('FWHM (\u03BCs)', fontsize=7)

        [ax.tick_params(axis='both', which='major', labelsize=5, length=1.25) for ax in [ax1, ax2, ax3]]

    def spike_data_figure_second_row(self):
        self.rows.append(GridSpecFromSubplotSpec(1, 2, subplot_spec=self.grid[1], width_ratios=[5, 3]))
        ps_plotter = self.initialize_plotter(PeriStimulusPlotter, PROPORTION_OPTS, FIGURE_1_OPTS, 2, 2, 0)
        ps_plotter.plot_groups_data()
        stats_plotter = self.initialize_plotter(GroupStatsPlotter, GROUP_STAT_OPTS, FIGURE_1_OPTS, 2, 1, 1)
        stats_plotter.plot_group_stats_data()

    def initialize_plotter(self, plotter_class, data_opts, graph_opts, rows, cols, position_in_row):
        plotter = plotter_class(self.experiment, self.data_type_context, self.neuron_type_context,
                                graph_opts=graph_opts, plot_type='gridspec_subplot')
        plotter.data_opts = data_opts
        gridspec = GridSpecFromSubplotSpec(rows, cols, subplot_spec=self.rows[1][position_in_row], hspace=0.7)
        invisible_ax = self.get_subplot_ax(self.rows[1][position_in_row], invisible=True)
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
