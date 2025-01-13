from copy import deepcopy

from k_onda.base import Base
from .layout_simplified import Layout
from k_onda.utils import recursive_update
from .partition_mixins import AestheticsMixin, LayerMixin, LabelMixin


class ProcessorConfig(Base):
    def __init__(self, executive_plotter, full_spec, layout=None, parent_processor=None, 
                 figure=None, division_info=None, index=None, aesthetics=None, layers=None):
        
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
        
        self.processor_classes = {
            'series': Series,
            'section': Section,
            'split': Split,
            'segment': Segment,
            'container': Container
        }
        
        
class Processor(Base, AestheticsMixin, LayerMixin, LabelMixin):
    def __init__(self, config):
        # Copy all attributes from config to the Processor instance
        self.__dict__.update(config.__dict__)

        if self.next:
            self.child_layout = Layout(self.parent_layout, self.current_index, processor=self, 
                                    figure=self.figure) 
        self.layers = self.init_layers()
        self.aesthetics = self.init_aesthetics()

    def next_processor_config(self, spec, updated_division_info):
        plot_type = spec.get('plot_type', self.plot_type) 
        cell = self.child_layout.cells[*self.current_index]

        return ProcessorConfig(
            self.executive_plotter, spec, layout=self.child_layout, 
            division_info=updated_division_info, figure=cell, 
            plot_type=plot_type, parent_processor=self, layers=self.layers)
        
    def start_next_processor(self, spec, updated_division_info):

        if 'calc_opts' in spec: 
            self.calc_opts = spec['calc_opts']
            self.experiment.initialize_data()

        config = self.next_processor_config(spec, updated_division_info)
        
        processor = self.processor_classes[config.spec_type](config)
        processor.start()


class Partition(Processor):

    def __init__(self, config):
        super().__init__(config)
        
        # a list of dictionaries with the unique combinations of values for the divisions
        self.info_by_division = []
        # self.info_by_division_by_layers is a list with these same unique values, repeated for
        # each unique layer
        self.info_dicts = self.info_by_division_by_layers if self.layers else self.info_by_division
        
        if 'label' in self.spec:
            self.label()

        self.assign_data_sources()

    def start(self):
        self.process_divisions(self.spec['divisions'])

    def assign_data_sources(self):
        for division in self.spec['divisions']:
            data_source = division.get('data_source')
            
            if not data_source:
                return
            # if data_source is 'all_animals' or similar, expand that into a list of identifiers
            if 'all' in division.get('members', []):
                division['members'] = [s.identifier for s in getattr(self.experiment, division['members'])]
            
    def process_divisions(self, divisions, info=None):
        """
        Recursively process a list of divider dicts, building up a cartesian product.
        Each divider dict looks like:
            {
                'divider_type': 'conditions',
                'members': [...],
                'dim': ...
            }
        `info` is the accumulated info from previous dividers.
        """
        if info is None:
            info = {}

        # we hit a leaf in the recursion
        if not divisions:
            # This is the final combination of all previous divider choices.
            self.info_by_division.append(info)            
            self.wrap_up(info)
            return

        # Otherwise, take the first divider in the list
        divider = divisions[0]
        divider_type = divider['divider_type']

        # Go through each of its members
        for i, member in enumerate(divider['members']):
            # Merge it into our accumulated info
            
            if not isinstance(member, str):
                updated_info = {**info, divider_type: info.get(divider_type, []) + [member]}
            else:
                updated_info = {**info, divider_type: member}

            self.advance_index(divider, i)

            # Now recurse on the remainder of the list, carrying `updated_info`
            self.process_divisions(divisions[1:], info=updated_info)

    def advance_index(self, current_divider, i):
        if self.name == 'segment':
            return
        if 'dim' in current_divider:
            dim = current_divider['dim']
            self.current_index[dim] = self.starting_index[dim] + i

    def get_calcs(self):
        
        info = self.info_by_division[-1]
        
        for key in ['neuron_type', 'period_type', 'period_group', 'period_types', 'conditions']:
           if key in info: 
                member = info[key]
                if isinstance(member, list):
                    val_to_set = {key: value for d in member for key, value in d.items()}
                else:
                    val_to_set = member
                setattr(self, f"selected_{key}", val_to_set)

        if 'data_source' not in info:
            data_source = self.experiment
        else:
            data_source = self.get_data_sources(data_object_type = info['data_object_type'], 
                                            identifier=info['data_source'])
        
        attr = self.spec.get('attr', 'calc')

        # TODO I need to move everything to do with layers in here.  I already
        # deleted it from ExecutivePlotter; it's not its job.
        if self.layers:
            self.get_layer_calcs()
        else:
            info.update({
                'attr': attr, 
                attr: getattr(data_source, attr), 
                data_source.name: data_source.identifier})
        

class Series(Partition):

    name = 'series'

    def __init__(self, config):
        super().__init__(config)
                
    def wrap_up(self, updated_info):
        
        for component in self.spec['components']:
            base = deepcopy(self.spec['base'])
            spec = recursive_update(base, component)
            self.start_next_processor(spec, updated_info) 
                

class Section(Partition):

    name = 'section'
    
    def __init__(self, config):
        super().__init__(config)

    def wrap_up(self, updated_info):
        
        cell = self.child_layout.cells[*self.current_index]

        if not self.next:
            self.get_calcs()
            self.executive_plotter.delegate(
                cell, info=[self.info_dicts.pop()], spec=self.spec, 
                aesthetics=self.aesthetics, is_last=not self.next)

        else:
            self.start_next_processor(self.next, updated_info)
          

class Segment(Partition):

    name = 'segment'

    def __init__(self, config):
        super().__init__(config)
        if not self.parent_layout:
            self.parent_layout = Layout(self.parent_layout, self.current_index, processor=self, 
                                    figure=self.figure)
            self.child_layout = self.parent_layout

    def start(self):
        super().start()
        cell = self.parent_layout.cells[*self.current_index]
        self.executive_plotter.delegate(
            cell, layout=self.parent_layout, info=self.info_by_division, spec=self.spec, 
            plot_type=self.plot_type, aesthetics=self.aesthetics, is_last=True)
        
    def wrap_up(self, _): 
        self.get_calcs()
           

class Split:
    pass


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

