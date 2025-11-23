import json
import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import product
from math import ceil

from .plotting_helpers import  PlottingMixin
from k_onda.core import OutputGenerator
from .processors.partitions import Section, Segment, Series, Split
from .processors.processor import Container, ProcessorConfig
from .processors.processor_mixins import MarginMixin
from .layout.layout import Layout
from .feature import (
    CategoricalScatterPlotter, LinePlotter, VerticalLinePlotter, BarPlotter, WaveformPlotter, CategoricalLinePlotter, 
    RasterPlotter, PeriStimulusHistogramPlotter, HeatMapPlotter, PeriStimulusHeatMapPlotter, 
    PeriStimulusPowerSpectrumPlotter, FeaturePlotter)
from k_onda.utils import to_serializable, PrepMethods, safe_make_dir, is_iterable




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
        self.layout = None
              
    def plot(self, opts):
        """
        The top-level method to plot data. Called by the Runner instance. Sets `calc_opts` and 
        initializes the experiment data, kicks off the top-level processor, and saves the figure to 
        disk.
        """

        if 'calc_opts' in opts:
            self.calc_opts = opts['calc_opts']
            self.experiment.initialize_data()

        self.process_plot_spec(opts)

    def count_axes(self, spec, rows=1, cols=1) -> tuple:
        """
        Walks the spec tree and returns an tuple representing how many rows and
        columns the layout will ultimately try to allocate.
        """
    
        if 'divisions' in spec:
            rows, cols = self.count(spec, rows, cols)

        for spec_type in ['series', 'section']:
            if spec_type in spec:
                rows, cols = self.count_axes(spec[spec_type], rows, cols)

        return rows, cols
    
    def count(self, spec, rows=1, cols=1):
      
        for division in spec['divisions']: # todo: containers don't have divisions
            dim = division.get('dim')
            members = division['members']
            if dim == 0:
                rows *= len(members)
            elif dim == 1:
                cols *= len(members)
            else: # dim wasn't specified by the user
                dimensions = division.get('dimensions', (4, 3))
                # this is going to build columnwise
                # the number of new rows is going to be members//max_cols
                # number of new cols is going to be min(len(members), max_cols)
                # todo add the ability to build rowwise first?
                rows *= ceil(len(members) / dimensions[1])
                cols *= min(len(members), dimensions[1])
        return rows, cols
        
    def spec_inspector(self, plot_spec, max_rows=4, max_cols=3):
      
        if 'split' in plot_spec:
            return self.make_split_combinations(plot_spec)
        
        if 'series' not in plot_spec:
            return [plot_spec]  # nothing obvious to split on

        rows, cols = self.count_axes(plot_spec['series'])
        if rows <= max_rows and cols <= max_cols:
            return [plot_spec]

        if rows - max_rows <= cols - max_cols:
            direction = 0
            mx = max_rows
        else:
            direction = 1
            mx = max_cols

        page_specs = self.expand_spec_into_splits(plot_spec, direction, mx)

        return page_specs

    
    def expand_spec_into_splits(self, ps, direction, mx):
        divisions = ps['series']['divisions']
        try:
            target_div = next(d for d in divisions if d.get('dim') == direction)
        except StopIteration:
            target_div = divisions[0]
        members = target_div['members']
        page_members = list(self.chunk_list(members, mx))
        spec = deepcopy(ps)
        series_spec = spec.pop('series')
        spec['split'] = {'divisions': [{'members': page_members}]}
        spec['split']['series'] = series_spec  # or however you want to embed it
        return self.make_split_combinations(spec)

    def chunk_list(self, seq, n):
        for i in range(0, len(seq), n):
            yield seq[i:i+n]
    
    def make_split_combinations(self, spec):
        # this gets a dictionary like {'split':{'divisions' = []}, 'section':{}}}
        bunches = []
        ranges = [range(len(division['members'])) for division in spec['split']['divisions']]
        all_combinations = list(product(*ranges))
        for combination in all_combinations:
            new_spec = deepcopy(spec)
            split_spec = new_spec.pop('split')
            new_spec['series'] = split_spec
            for i, j in enumerate(combination):
                new_members = new_spec['series']['divisions'][i]['members'][j]
                if not is_iterable(new_members):
                    new_members = [new_members]
                new_spec['series']['divisions'][i]['members'] = new_members
                 
            bunches.append(new_spec)
        return bunches

    def process_plot_spec(self, opts):
        """
        Make a figure and start the top-level processor.
        """

        plot_spec = opts['plot_spec']

        # apply pagination / auto-split
        pages = self.spec_inspector(plot_spec, max_rows=4, max_cols=3)

        for spec in pages:
            self.kick_off(spec)
            self.wrap_up(opts)

    def kick_off(self, spec):

        processor_classes = {
            'split': Split,
            'section': Section,
            'segment': Segment,
            'series': Series,
            'container': Container
        }

        self.make_fig(spec)
     
        config = ProcessorConfig(self, spec, layout=self.layout, 
                                  figure=self.layout.cells[0, 0], index=[0, 0], 
                                 is_first=True)
        processor = processor_classes[config.spec_type](config)
        processor.start(top_level=True)

    def make_fig(self, plot_spec):
        """
        Make and return a figure and the top-level layout.
        """     

        if plot_spec.get('figsize'):
            self.fig = plt.figure(figsize=plot_spec['figsize'])
        else:
            self.fig = plt.figure()
        if plot_spec.get('subplots_adjust'):
            self.fig.subplots_adjust(**plot_spec['subplots_adjust'])
        if plot_spec.get('margins'):
            gs_args = self.calculate_margins(plot_spec['margins'])
        else:
            gs_args = {}
        self.layout = Layout(None, [0, 0], figure=self.fig, gs_args=gs_args)
    
    def get_margins_from_spec(self, spec):
        for k in ['series', 'section', 'segment', 'container']:
            if k in spec:
                return spec[k].get('margins', {})
    
    def wrap_up(self, opts):
        interactive = opts.get('interactive', False)
        if interactive:
             plt.show()
        self.close_plot(opts=opts)

    def close_plot(self, opts=None, fig=None):
        
        if not fig:
            fig = self.fig  

        self.build_write_path(opts=opts)

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
        send_args = dict(
            spec=spec,
            plot_type=plot_type,
            aesthetics=aesthetics,
            ax=ax,
            spec_type=spec_type,
            legend_info_list=legend_info_list
        )

        if info and isinstance(info, list) and any('split' in entry for entry in info):
            splits = {entry['split'] for entry in info}
            self.make_fig(spec)
            for split in splits:
                subset = [entry for entry in info if entry['split'] == split]
                self.send(info=subset, **send_args)
        else:
            self.send(info=info, **send_args)


    def send(self, info=None, spec=None, plot_type=None, aesthetics=None, ax=None,
                  spec_type=None, legend_info_list=None):
        calc_config = dict(info=info, spec=spec, plot_type=plot_type, aesthetics=aesthetics, ax=ax,
                  spec_type=spec_type, legend_info_list=legend_info_list)
        if plot_type in PLOT_TYPES:
            PLOT_TYPES[plot_type]().process_calc(calc_config)
        else:
            FeaturePlotter().process_calc(calc_config)
        
