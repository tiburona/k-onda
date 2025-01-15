from copy import deepcopy

from k_onda.base import Base
from ..layout import Layout
from .partition_mixins import AestheticsMixin, LayerMixin, MarginMixin, LabelMixin


class ProcessorConfig(Base):
    """
    Creates the configuration object for each processor.  Its purpose is to avoid long, repetitive
    __init__ methods for each processor subclass.
    """

    def __init__(self, executive_plotter, full_spec, layout=None, parent_processor=None, 
                 figure=None, division_info=None, index=None, aesthetics=None, layers=None, 
                 is_first=False):
        
        self.executive_plotter = executive_plotter
        self.full_spec = full_spec
        processor_types = ['section', 'split', 'segment', 'series', 'container']
        self.spec_type = [k for k in processor_types if k in self.full_spec][0]
        self.spec = self.full_spec[self.spec_type]
        self.parent_layout = layout
        self.parent_processor = parent_processor
        self.figure = figure
        self.division_info = division_info
        self.index = index
        self.aesthetics = aesthetics
        self.layers = layers
        self.is_first = is_first
        self.plot_type = self.full_spec.get('plot_type')
        self.next = None
        for k in processor_types:
            if k in self.spec:
                self.next = {k: self.spec[k]}
        
        if self.index:
            self.starting_index = self.index
        else:
            self.starting_index = [0, 0]

        self.inherited_division_info = self.division_info if self.division_info else {}  
        self.current_index = deepcopy(self.starting_index)
        
class Processor(Base, LayerMixin, AestheticsMixin, LabelMixin, MarginMixin):
    """
    The base class for all processors. Its responsible for initializing the `child_layout` and 
    configuring and starting the next processor, if it exists.
    """

    def __init__(self, config):
        self.__dict__.update(config.__dict__)
        
        self.child_layout = Layout(
            self.parent_layout,
            self.current_index,
            processor=self,
            figure=self.figure,
            **self.get_layout_args()  # Dynamically include additional arguments
        )
        
        self.layers = self.init_layers()
        self.aesthetics = self.init_aesthetics()
        self.label()

    def get_layout_args(self):
        """Provides additional arguments for Layout."""
        return {}  # Default implementation provides no extra arguments
        
    def next_processor_config(self, spec, updated_division_info):
        plot_type = spec.get('plot_type', self.plot_type) 
        cell = self.child_layout.cells[*self.current_index]

        return ProcessorConfig(
            self.executive_plotter, spec, layout=self.child_layout, 
            division_info=updated_division_info, figure=cell, 
            plot_type=plot_type, parent_processor=self, layers=self.layers)
        
    def start_next_processor(self, spec, updated_division_info):

        from .processor_map import PROCESSOR_MAP as processor_map

        if 'calc_opts' in spec: 
            self.calc_opts = spec['calc_opts']
            self.experiment.initialize_data()

        config = self.next_processor_config(spec, updated_division_info)
        
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
                    self.start_next_processor(spec, self.inherited_info)

                elif kind == 'text':
                    current_cell = self.child_layout[*self.current_index]
                    ax = self.child_layout.add_ax(current_cell, (i, j))
                    self.executive_plotter.delegate(ax, spec=spec)

                elif kind == 'image':
                    pass

                    

                # else: handle table, or skip if empty, etc.

