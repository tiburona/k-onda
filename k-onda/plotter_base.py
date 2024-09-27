from base_data import Base

class PlotterBase(Base):
    _experiment = None
    _origin_plotter = None
    _active_plotter = None
    _active_figurer = None
    _active_ax = None
    _active_spec_type = None
    _active_spec = None

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
    def active_figurer(self):
        return PlotterBase._active_figurer

    @active_figurer.setter
    def active_figurer(self, value):
        PlotterBase._active_figurer = value

    @property
    def active_ax(self):
        return PlotterBase._active_ax

    @active_ax.setter
    def active_ax(self, value):
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

