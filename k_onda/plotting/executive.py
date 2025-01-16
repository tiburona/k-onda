import os
import json
import uuid

import matplotlib.pyplot as plt

from .plotting_helpers import smart_title_case, PlottingMixin
from k_onda.base import Base
from .processors.partitions import Section, Segment, Series
from .processors.processor import Container, ProcessorConfig
from .processors.partition_mixins import MarginMixin
from .layout import Layout
from .feature import (
    CategoricalScatterPlotter, LinePlotter, BarPlotter, WaveformPlotter, CategoricalLinePlotter, 
    RasterPlotter, PeriStimulusHistogramPlotter, HeatMapPlotter)
from k_onda.utils import to_serializable, PrepMethods


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial'] 


PLOT_TYPES = {'categorical_scatter': CategoricalScatterPlotter,
              'line_plot': LinePlotter,
              'bar_plot': BarPlotter,
              'waveform': WaveformPlotter,
              'categorical_line': CategoricalLinePlotter,
              'raster': RasterPlotter,
              'psth': PeriStimulusHistogramPlotter,
              'heat_map': HeatMapPlotter}  


class ExecutivePlotter(Base, PlottingMixin, PrepMethods, MarginMixin):
    """
    The class to orchestrate the plotting of data. Sets `calc_opts` and initializes the 
    experiment data, creates the figure, starts the top-level processor, upon completion of the last 
    processor delegates to the appropriate feature plotter, and saves the figure to disk.
    """

    def __init__(self, experiment):
        self.experiment = experiment
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
        self.write_opts = opts.get('write_opts', {}) 
        plot_spec = opts['plot_spec']
        self.process_plot_spec(plot_spec)
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
        processor.start()

    def make_fig(self, plot_spec):
        """
        Make and return a figure and the top-level layout.
        """     

        self.fig = plt.figure(layout='tight')
        if plot_spec.get('margins'):
            gs_args = self.calculate_margins(plot_spec['margins'])
        else:
            gs_args = {}
        self.layout = Layout(self, [0, 0], figure=self.fig, gs_args=gs_args)
        return self.fig
    
    def get_margins_from_spec(self, spec):
        for k in ['series', 'section', 'segment', 'container']:
            if k in spec:
                return spec[k].get('margins', {})

    def construct_path(self):
        # Fill fields
        root, fname, path = [
            self.fill_fields(self.write_opts.get(key)) 
            for key in ['root', 'fname', 'path']
        ]
        
        # If user explicitly set 'path', we skip building our own path
        if path:
            self.file_path = path
    
        else:
            # Fallback to some default root
            if not root:
                root = self.experiment.exp_info.get('data_path', os.getcwd())
            
            # Fallback to some default fname
            if not fname:
                fname = '_'.join([self.kind_of_data, self.calc_type])
            
            self.title = smart_title_case(fname.replace('_', ' '))
            
            # Build partial path (no extension yet)
            self.file_path = os.path.join(root, self.kind_of_data, fname)

        self.handle_collisions()
        
        # Build final file path and an opts file
        ext = self.write_opts.get('extension', '.png')
        self.opts_file_path = self.file_path + '.txt'
        self.file_path += ext

    def handle_collisions(self):
        if os.path.exists(self.file_path) and not self.write_opts.get('allow_overwrite', True):
            uuid_str = str(uuid.uuid4())[:self.write_opts.get('unique_hash', 8)]
            basename = os.path.basename(self.file_path)
            dir_ = os.path.dirname(self.file_path)
            new_name = f"{basename}_{uuid_str}"
            self.file_path = os.path.join(dir_, new_name)

    def close_plot(self, basename='', fig=None):
        
        if not fig:
            fig = self.fig  
        plt.show()
        self.save_and_close_fig(fig, basename)
       
    def save_and_close_fig(self, fig, do_title=True):
        
        self.construct_path()

        if do_title:
            bbox = fig.axes[0].get_position()
            fig.suptitle(self.title, fontsize=16, y=bbox.ymax + 0.1)
       
        fig.savefig(self.file_path, bbox_inches='tight', dpi=300)

        with open(self.opts_file_path, 'w') as file:
            json.dump(to_serializable(self.calc_opts), file)

        plt.close(fig)

    def delegate(self, info=None, spec=None, plot_type=None, aesthetics=None, ax=None,
                  spec_type=None):
        """
        Delegate to the appropriate feature plotter based on the plot_type.
        """
        if spec_type == 'container':
            # Containers pass different arguments to delegate. They have no info, only an ax.
            PLOT_TYPES[plot_type]().process_calc(ax, info, spec=spec)
        else:
            PLOT_TYPES[plot_type]().process_calc(info, spec, spec_type, aesthetics=aesthetics)
            

        
        

        