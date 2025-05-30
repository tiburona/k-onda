import json
import matplotlib.pyplot as plt

from .plotting_helpers import  PlottingMixin
from k_onda.core import OutputGenerator
from .processors.partitions import Section, Segment, Series
from .processors.processor import Container, ProcessorConfig
from .processors.processor_mixins import MarginMixin
from .layout import Layout
from .feature import (
    CategoricalScatterPlotter, LinePlotter, VerticalLinePlotter, BarPlotter, WaveformPlotter, CategoricalLinePlotter, 
    RasterPlotter, PeriStimulusHistogramPlotter, HeatMapPlotter, PeriStimulusHeatMapPlotter, 
    PeriStimulusPowerSpectrumPlotter)
from k_onda.utils import to_serializable, PrepMethods, safe_make_dir


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial'] 


PLOT_TYPES = {'categorical_scatter': CategoricalScatterPlotter,
              'line_plot': LinePlotter,
              'vertical_line': VerticalLinePlotter,
              'bar_plot': BarPlotter,
              'waveform': WaveformPlotter,
              'categorical_line': CategoricalLinePlotter,
              'raster': RasterPlotter,
              'psth': PeriStimulusHistogramPlotter,
              'heat_map': HeatMapPlotter,
              'peristimulus_heat_map': PeriStimulusHeatMapPlotter,
              'peristimulus_power_spectrum': PeriStimulusPowerSpectrumPlotter}  


class ExecutivePlotter(OutputGenerator, PlottingMixin, PrepMethods, MarginMixin):
    """
    The class to orchestrate the plotting of data. Sets `calc_opts` and initializes the 
    experiment data, creates the figure, starts the top-level processor, upon completion of the last 
    processor delegates to the appropriate feature plotter, and saves the figure to disk.
    """

    def __init__(self, experiment):
        super().__init__()
        self.experiment = experiment
        self.io_opts = None
        self.write_opts = None
              
    def plot(self, opts):
        """
        The top-level method to plot data. Called by the Runner instance. Sets `calc_opts` and 
        initializes the experiment data, kicks off the top-level processor, and saves the figure to 
        disk.
        """
        if 'calc_opts' in opts:
            self.calc_opts = opts['calc_opts']
            self.experiment.initialize_data()

        plot_spec = opts['plot_spec']
        self.process_plot_spec(plot_spec)
        interactive = opts.get('interactive', False)
        if interactive:
            plt.show()
        self.close_plot(opts.get('fname', ''))

    def process_plot_spec(self, plot_spec):
        """
        Make a figure and start the top-level processor.
        """

        processor_classes = {
            'section': Section,
            'segment': Segment,
            'series': Series,
            'container': Container
        }
    
        self.make_fig(plot_spec)
    
        config = ProcessorConfig(self, plot_spec, layout=self.layout, 
                                  figure=self.layout.cells[0, 0], index=[0, 0], 
                                 is_first=True)
        processor = processor_classes[config.spec_type](config)
        processor.start(top_level=True)

    def make_fig(self, plot_spec):
        """
        Make and return a figure and the top-level layout.
        """     

        self.fig = plt.figure()
        self.fig.subplots_adjust()
        if plot_spec.get('margins'):
            gs_args = self.calculate_margins(plot_spec['margins'])
        else:
            gs_args = {}
        self.layout = Layout(self, [0, 0], figure=self.fig, gs_args=gs_args)
    
    def get_margins_from_spec(self, spec):
        for k in ['series', 'section', 'segment', 'container']:
            if k in spec:
                return spec[k].get('margins', {})

    def close_plot(self, basename='', fig=None):
        
        if not fig:
            fig = self.fig  
        self.save_and_close_fig(fig, basename)
       
    def save_and_close_fig(self, fig, do_title=True):
        
        self.build_write_path()

        if do_title:
            bbox = fig.axes[0].get_position()
            fig.suptitle(self.title, fontsize=16, y=bbox.ymax + 0.1)
        
        safe_make_dir(self.file_path)
        
        fig.savefig(self.file_path, bbox_inches='tight', dpi=300)

        with open(self.opts_file_path, 'w') as file:
            json.dump(to_serializable(self.calc_opts), file)

        plt.close(fig)

    def delegate(self, info=None, spec=None, plot_type=None, aesthetics=None, ax=None,
                  spec_type=None, legend_info_list=None):
        """
        Delegate to the appropriate feature plotter based on the plot_type.
        """
        calc_config = dict(info=info, spec=spec, plot_type=plot_type, aesthetics=aesthetics, ax=ax,
                  spec_type=spec_type, legend_info_list=legend_info_list)
        PLOT_TYPES[plot_type]().process_calc(calc_config)
