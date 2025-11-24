import json
import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import product
from math import ceil

from .plotting_helpers import  PlottingMixin
from k_onda.core import OutputGenerator
from .processors.partitions import Section, Segment, Series
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
        Walks the spec tree and returns (rows, cols) for the layout,
        ignoring any dim==2 (page axes).
        """
        if 'divisions' in spec:
            rows, cols = self.count(spec, rows, cols)

        for spec_type in ['series', 'section']:
            if spec_type in spec:
                rows, cols = self.count_axes(spec[spec_type], rows, cols)

        return rows, cols

    def count(self, spec, rows=1, cols=1):
        for division in spec['divisions']:
            dim = division.get('dim')
            members = division['members']

            if dim == 0:
                rows *= len(members)
            elif dim == 1:
                cols *= len(members)
            elif dim == 2:
                # page axis: DO NOT affect rows/cols
                continue
            else:
                dimensions = division.get('dimensions', (4, 3))
                rows *= ceil(len(members) / dimensions[1])
                cols *= min(len(members), dimensions[1])

        return rows, cols
    
    def descend_spec(self, spec, fun):
        # apply to this node if appropriate
        if 'divisions' in spec:
            fun(spec)

        for key in ('series', 'section', 'segment'):
            child = spec.get(key)
            if child is not None:
                self.descend_spec(child, fun)


    def assign_data_sources(self, spec):
        divisions = spec.get('divisions', [])
        for division in divisions:
            members = division.get('members')

            data_source = division.get('data_source')
            if data_source is None:
                continue

            if members == 'all' or members == ['all']:
                pool = getattr(self.experiment, f'all_{data_source}s')
                division['members'] = [obj.unique_id for obj in pool]
                continue

    def spec_inspector(self, plot_spec, max_rows=4, max_cols=3):
        """
        Decide whether to paginate and return a list of page specs.

        - If a division with dim==2 exists in the top partition (series/section),
        paginate explicitly along that axis.
        - Otherwise, if the layout overflows max_rows/max_cols, do a rescue
        pagination by chopping one division into pages.
        """
        self.descend_spec(plot_spec, self.assign_data_sources)

        # Which partition are we operating on? series preferred, else section.
        part_key = None
        for key in ('series', 'section'):
            if key in plot_spec:
                part_key = key
                break

        if part_key is None:
            return [plot_spec]  # nothing obvious to paginate

        part_spec = plot_spec[part_key]
        divisions = part_spec['divisions']

        # 1) Explicit page axis: dim == 2
        for idx, d in enumerate(divisions):
            if d.get('dim') == 2:
                dimensions = d.get('dimensions', (4, 3))
                per_page = dimensions[0] * dimensions[1]
                return self._paginate_over_division(plot_spec, part_key, idx, per_page)

        # 2) No explicit page axis: compute layout and maybe rescue
        rows, cols = self.count_axes(part_spec)
        if rows <= max_rows and cols <= max_cols:
            return [plot_spec]

        row_over = max(rows - max_rows, 0)
        col_over = max(cols - max_cols, 0)

        if row_over >= col_over:
            target_dim = 0
            page_size = max_rows
        else:
            target_dim = 1
            page_size = max_cols

        # find division with that dim; fallback to first
        try:
            div_idx = next(i for i, d in enumerate(divisions)
                        if d.get('dim') == target_dim)
        except StopIteration:
            div_idx = 0

        return self._paginate_over_division(plot_spec, part_key, div_idx, page_size)
    
    def _paginate_over_division(self, plot_spec, part_key, div_idx, page_size):
        """
        Given a plot_spec and a partition key ('series' or 'section'),
        split one division's members into pages and return per-page specs.
        """
        pages = []
        divisions = plot_spec[part_key]['divisions']
        members = divisions[div_idx]['members']
        if isinstance(members, str):
             members = [members]

        for chunk in self.chunk_list(members, page_size):
            spec = deepcopy(plot_spec)
            spec[part_key]['divisions'][div_idx]['members'] = chunk
            pages.append(spec)

        return pages

    def chunk_list(self, seq, n):
        for i in range(0, len(seq), n):
            yield seq[i:i+n]

    def process_plot_spec(self, opts):
        plot_spec = opts['plot_spec']
        pages = self.spec_inspector(plot_spec, max_rows=4, max_cols=3)

        for spec in pages:
            self.kick_off(spec)
            self.wrap_up(opts)


    def kick_off(self, spec):

        processor_classes = {
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

        self.send(info=info, **send_args)

    def send(self, info=None, spec=None, plot_type=None, aesthetics=None, ax=None,
                  spec_type=None, legend_info_list=None):
        calc_config = dict(info=info, spec=spec, plot_type=plot_type, aesthetics=aesthetics, ax=ax,
                  spec_type=spec_type, legend_info_list=legend_info_list)
        if plot_type in PLOT_TYPES:
            PLOT_TYPES[plot_type]().process_calc(calc_config)
        else:
            FeaturePlotter().process_calc(calc_config)
        
