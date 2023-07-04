import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np

from plotters import Plotter
from phy_interface import PhyInterface
from initialize_experiment import experiment, data_type_context, neuron_type_context
from opts_library import GROUP_STAT_OPTS, PROPORTION_OPTS, FIGURE_1_OPTS

plt.rcParams['font.family'] = 'Arial'


class Figure:

    def __init__(self, exp, dt_context, nt_context, graph_opts):
        self.fig = None
        self.experiment = exp
        self.data_type_context = dt_context
        self.neuron_type_context = nt_context
        self.graph_opts = graph_opts
        self.fig = plt.figure()
        self.grid = GridSpec(2, 1)
        self.rows = []

    def spike_data_figure(self):
        self.grid.update(hspace=0.5)
        self.spike_data_figure_first_row()
        self.spike_data_figure_second_row()
        plt.show()

    def spike_data_figure_first_row(self):

        self.rows.append(GridSpecFromSubplotSpec(1, 3, subplot_spec=self.grid[0], width_ratios=[3, 3, 4]))
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
            ax1.scatter(x, y, alpha=0.3, color=colors[i])
            waveform = phy_interface.get_mean_waveforms(cluster_id, electrodes)
            ax2.plot(np.arange(len(waveform)), waveform, color=colors[i])

    def pn_in_scatterplot(self):
        scatterplot = GridSpecFromSubplotSpec(3, 3, subplot_spec=self.rows[0][2])

        # Large subplot in latter two columns and first two rows
        ax1 = self.get_subplot_ax(scatterplot[:2, 1:])
        x = [unit.fwhm_microseconds for unit in self.experiment.all_units]
        y = [unit.firing_rate for unit in self.experiment.all_units]

        colors = [self.graph_opts['neuron_type_colors'][unit.neuron_type] for unit in self.experiment.all_units]
        ax1.scatter(x, y, color=colors, alpha=0.5)  # Scatter plot on ax1
        ax1.text(0.5, 0.5, "", ha='center')

        # Small subplot in the first column and first two rows
        ax2 = self.get_subplot_ax(scatterplot[:2, 0])
        ax2_twin = ax2.twinx()
        ax2_twin.hist(y, bins=30, orientation='horizontal', color='purple', alpha=0.5)
        ax2_twin.set_xlim(ax2_twin.get_xlim()[::-1])  # Flip the x-axis
        ax2_twin.yaxis.set_ticks([])  # Hide the y-axis ticks on the twin axis
        ax2.axis('off')  # Hide the original axes

        # Small subplot in the last row and latter two columns
        ax3 = self.get_subplot_ax(scatterplot[2, 1:])
        ax3.hist(x, bins=30, color='purple', alpha=0.5)  # Histogram on ax2
        ax3.text(0.5, 0.5, "", ha='center')

    def spike_data_figure_second_row(self):
        self.rows.append(GridSpecFromSubplotSpec(1, 2, subplot_spec=self.grid[1], width_ratios=[6, 3]))
        plotter = Plotter(self.experiment, self.data_type_context, self.neuron_type_context, graph_opts=FIGURE_1_OPTS,
                          plot_type='gridspec_subplot')
        invisible_axs = [self.get_subplot_ax(self.rows[1][subplot], invisible=True) for subplot in range(2)]

        plotter.data_opts = PROPORTION_OPTS
        group_plot = GridSpecFromSubplotSpec(2, 2, subplot_spec=self.rows[1][0], hspace=1.0)
        plotter.set_gridspec_axes(self.fig, group_plot, 2, 2, invisible_ax=invisible_axs[0])
        plotter.plot_groups_data()

        plotter.data_opts = GROUP_STAT_OPTS
        post_hoc_plot = GridSpecFromSubplotSpec(2, 1, subplot_spec=self.rows[1][1], hspace=1.0)
        plotter.set_gridspec_axes(self.fig, post_hoc_plot, 2, 1, invisible_ax=invisible_axs[1])
        plotter.plot_group_stats_data()

    def get_subplot_ax(self, gridspec_slice, invisible=False):
        ax1 = plt.Subplot(self.fig, gridspec_slice)
        self.fig.add_subplot(ax1)
        if invisible:
            ax1.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
            for position in ['top', 'right', 'bottom', 'left']:
                ax1.spines[position].set_visible(False)
        return ax1


figure = Figure(experiment, data_type_context, neuron_type_context, FIGURE_1_OPTS)
figure.spike_data_figure()
