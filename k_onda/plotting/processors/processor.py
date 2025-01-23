from copy import deepcopy, copy

from k_onda.base import Base
from ..layout import Layout
from .processor_mixins import AestheticsMixin, LayerMixin, MarginMixin, LabelMixin, LegendMixin
from ..plotting_helpers import PlottingMixin


class ProcessorConfig(Base):
    """
    Creates the configuration object for each processor.  Its purpose is to avoid long, repetitive
    __init__ methods for each processor subclass.
    """

    def __init__(self, executive_plotter, full_spec, layout=None, parent_processor=None, 
                 figure=None, division_info=None, info_by_division=None, index=None, aesthetics=None, layers=None, 
                 is_first=False, plot_type=None, legend_info_list=None):
        
        self.executive_plotter = executive_plotter
        self.full_spec = full_spec
        processor_types = ['section', 'split', 'segment', 'series', 'container']
        self.spec_type = [k for k in processor_types if k in self.full_spec][0]
        self.spec = self.full_spec[self.spec_type]
        self.parent_layout = layout
        self.parent_processor = parent_processor
        self.figure = figure
        self.division_info = division_info
        self.info_by_division = info_by_division
        self.index = index
        self.aesthetics = aesthetics    
        self.layers = layers
        self.is_first = is_first
        self.label = None
        self.plot_type = plot_type or self.full_spec.get('plot_type')
        self.next = None
        for k in processor_types:
            if k in self.spec:
                self.next = {k: self.spec[k]}
        
        
        if self.index:
            self.starting_index = self.index
        else:
            self.starting_index = [0, 0]

        self.inherited_division_info = self.division_info if self.division_info is not None else {}
        self.info_by_division = self.info_by_division if self.info_by_division is not None else []
        self.legend_info_list = legend_info_list if legend_info_list is not None else []
        self.current_index = copy(self.starting_index)
        
class Processor(Base, PlottingMixin, LayerMixin, AestheticsMixin, LabelMixin, MarginMixin, LegendMixin):
    """
    The base class for all processors. Its responsible for initializing the `child_layout` and 
    configuring and starting the next processor, if it exists.
    """

    def __init__(self, config):
        self.__dict__.update(config.__dict__)

        self.layers = self.init_layers()
        self.aesthetics = self.init_aesthetics()
        self.label = self.get_label()

        self.child_layout = Layout(
            self.parent_layout,
            self.current_index,
            processor=self,
            figure=self.figure,
            **self.get_layout_args()  # Dynamically include additional arguments
        )
        
    def get_layout_args(self):
        """Provides additional arguments for Layout."""
        return {}  # Default implementation provides no extra arguments
        
    def next_processor_config(self, spec, updated_division_info, info_by_division):
        cell = self.child_layout.cells[*self.current_index]

        processor_config = dict(
            figure=cell, 
            layers=self.layers, 
            parent_processor=self, 
            plot_type=self.plot_type,
            layout=self.child_layout, 
            division_info=updated_division_info, 
            info_by_division=info_by_division,
            legend_info_list=self.legend_info_list
            )
        
        return ProcessorConfig(self.executive_plotter, spec, **processor_config)
        
    def start_next_processor(self, spec, updated_division_info, info_by_division):

        from .processor_map import PROCESSOR_MAP as processor_map

        if 'calc_opts' in spec: 
            self.calc_opts = spec['calc_opts']
            self.experiment.initialize_data()

        config = self.next_processor_config(spec, updated_division_info, info_by_division)
        
        processor = processor_map[config.spec_type](config)
        processor.start()


class Container(Processor):
    """
    A freeform container that can display text, images, or partitions in each cell.
    """

    def __init__(self, config):
        super().__init__(config)
        
    def check_type(self, spec):
        processor_keys = ['series', 'section', 'segment', 'split', 'container']
        for key in processor_keys:
            if key in spec:
                return 'processor'  
        return spec.get('type')  # e.g. 'text', 'image', or None
  
    def start(self):
        for i in range(self.child_layout.dimensions[0]):
            for j in range(self.child_layout.dimensions[1]):

                self.current_index = [i, j]
                spec = self.spec['components'][i][j]  # e.g. { 'type': 'text', 'content': 'Hello' }
                kind = self.check_type(spec)

                if kind == 'processor':
                    self.start_next_processor(spec, self.inherited_info, self.info_by_division)

                else:
                    current_cell = self.child_layout[*self.current_index]
                    ax = self.child_layout.add_ax(current_cell, (i, j))
                    self.executive_plotter.delegate(
                        plot_type=spec['plot_type'], info=self.inherited_info, spec=spec, ax=ax)

              

