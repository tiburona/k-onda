from ast import Sub
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from plotter_base import PlotterBase
from plotters import Figurer, ExecutivePlotter
from subplotter import Subplotter, Figurer




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
            plot_type, plot_spec, calc_opts, xy = (
                component.get(k) for k in ['plot_type', 'plot_spec', 'calc_opts', 'gs_xy'])
            self.active_fig = self.fig.add_subfigure(self.subplotter.gs[*(slice(*d) for d in xy)])
            
            self.subplotter.mark_edges_of_component(xy)
            graph_opts = {'plot_spec': plot_spec, 'graph_dir': self.layout_spec['graph_dir'], 
                          'plot_type': plot_type}
            plotter = ExecutivePlotter(self.experiment)
            plotter.plot(calc_opts=calc_opts, graph_opts=graph_opts, parent_figure=self.active_fig,
                         index=[xy[0][0], xy[1][0]])
            self.active_aesthetics = None

        basename = self.layout_spec.get('figure_name', 'figure')
        plotter.close_plot(basename=basename, fig=self.fig, do_title=False)
            


# what is a component?
# it can have a single constituent calc_opts or a series of them
# if it has a series of them how is that defined?
# what information do we need?
# a gridspec, and the relationship of the calc opts to rows and cols
# basically a component should be a list of calc opts, and if there's more than one,
# an optional   