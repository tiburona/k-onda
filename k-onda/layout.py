from ast import Sub
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from plotter_base import PlotterBase
from plotters import Figurer, CategoricalScatterPlotter, PeriStimulusHistogramPlotter, WaveformPlotter
from subplotter import Subplotter, Figurer


PLOTTING_CLASSES = {
    'categorical_scatter': CategoricalScatterPlotter,
    'waveform': WaveformPlotter,
}


class Layout(PlotterBase):
    
    def __init__(self, experiment):
        self.experiment = experiment
    
    def initialize(self, layout_spec):
        self.layout_spec = layout_spec
        self.is_layout = True
        self.figurer = Figurer()
        self.fig = self.figurer.make_fig()
        self.gs = self.figurer.gs
        self.subplotter = Subplotter(
            self, index=[0, 0], dimensions=self.layout_spec['dimensions'], 
            grid_keywords=self.layout_spec.get('grid_args', {}), 
            invisible_axes=self.layout_spec.get('invisible_axes', []))

    def make_figure(self, layout_spec):
        self.initialize(layout_spec)
    
        for component in self.layout_spec['components']:
            self.active_plotter = self.subplotter
            plot_cls, plot_spec, calc_opts, xy = (
                component[k] for k in ['plot_type', 'plot_spec', 'calc_opts', 'gs_xy'])
            self.active_fig = self.fig.add_subfigure(self.subplotter.gs[*(slice(*d) for d in xy)])
            # Call the plot function
            plotter = PLOTTING_CLASSES[plot_cls](self.experiment)
            graph_opts = {'plot_spec': plot_spec, 'graph_dir': self.layout_spec['graph_dir']}
            plotter.plot(calc_opts=calc_opts, graph_opts=graph_opts, parent_figure=self.active_fig,
                         index=[xy[0][0], xy[1][0]])

        basename = self.layout_spec.get('figure_name', 'figure')
        plotter.close_plot(basename=basename, fig=self.fig, do_title=False)
            


