from base_data import Base

class PlotterBase(Base):
    _experiment = None
    _origin_plotter = None
    _active_plotter = None
    _active_fig = None
    _active_ax = None
    _active_spec_type = None
    _active_spec = None
    _active_plot_type = None
    _active_aesthetics = None
    _is_layout = False

    @property
    def experiment(self):
        return PlotterBase._experiment

    @experiment.setter
    def experiment(self, value):
        PlotterBase._experiment = value

    @property
    def origin_plotter(self):
        return PlotterBase._origin_plotter

    @origin_plotter.setter
    def origin_plotter(self, value):
        PlotterBase._origin_plotter = value

    @property
    def active_plotter(self):
        return PlotterBase._active_plotter

    @active_plotter.setter
    def active_plotter(self, value):
        PlotterBase._active_plotter = value

    @property
    def active_fig(self):
        return PlotterBase._active_fig

    @active_fig.setter
    def active_fig(self, value):
        PlotterBase._active_fig = value

    @property
    def active_acks(self):
        return PlotterBase._active_ax

    @active_acks.setter
    def active_acks(self, value):
        PlotterBase._active_ax = value

    @property
    def active_spec_type(self):
        return PlotterBase._active_spec_type

    @active_spec_type.setter
    def active_spec_type(self, value):
        PlotterBase._active_spec_type = value

    @property
    def active_spec(self):
        return PlotterBase._active_spec

    @active_spec.setter
    def active_spec(self, value):
        PlotterBase._active_spec = value

    @property
    def is_layout(self):
        return PlotterBase._is_layout
    
    @is_layout.setter
    def is_layout(self, value):
        PlotterBase._is_layout = value

    @property
    def active_plot_type(self):
        return PlotterBase._active_plot_type
    
    @active_plot_type.setter
    def active_plot_type(self, value):
        PlotterBase._active_plot_type = value

    @property
    def active_aesthetics(self):
        return PlotterBase._active_aesthetics
    
    @active_aesthetics.setter
    def active_aesthetics(self, value):
        PlotterBase._active_aesthetics = value
    


