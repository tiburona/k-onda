from copy import deepcopy

from .plotter_base import PlotterBase
from .layout_simplified import Layout
from k_onda.utils import recursive_update
from .partition_mixins import AestheticMixin, LayerMixin


class ProcessorConfig:
    def __init__(self, executive_plotter, layout=None, parent_partition=None, figure=None, 
                 spec=None, division_info=None, plot_type=None, index=None,   ):
        self.executive_plotter = executive_plotter
        self.parent_layout = layout
        self.parent_partition = parent_partition
        self.figure = figure
        self.spec = spec
        self.division_info = division_info
        self.plot_type = plot_type
        self.index = index
        
        
class Processor(PlotterBase, AestheticMixin, LayerMixin):
    def __init__(self, config):
        super().__init__(config)
        
        self.inherited_division_info = self.info if self.info else {}  

        if self.index:
            self.starting_index = self.index
        else:
            self.starting_index = [0, 0]
        self.current_index = deepcopy(self.starting_index)

        self.next = None
        for k in ('series', 'section', 'segment', 'components', 'container'):
            if k in self.spec:
                self.next = {k: self.spec[k]}

        if self.next:
            self.child_layout = Layout(self.parent_layout, self.current_index, processor=self, 
                                    figure=self.figure)
            
        self.layers = self.init_layers()
        self.aesthetic = self.init_aesthetic()
            
        self.processor_classes = {
            'series': Series,
            'section': Section,
            'segment': Segment,
            'split': Split,
            'container': Container
        }

    def next_processor_config(self, spec, updated_division_info):
        spec_type, spec = list(spec.items())[0]
        plot_type = spec.pop('plot_type', None) or self.plot_type 
        cell = self.child_layout.cells[*self.current_index]

        config = ProcessorConfig(
            self.executive_plotter, layout=self.child_layout, division_info=updated_division_info, 
            figure=cell, spec=spec, plot_type=plot_type)
        
        return spec_type, config

    def start_next_processor(self, spec, updated_division_info):

        if 'calc_opts' in spec: 
            self.calc_opts = spec.pop('calc_opts')
            self.experiment.initialize_data()

        spec_type, config = self.next_processor_config(spec, updated_division_info)

        if spec_type == 'segment':
            raise ValueError("Segments should be the final processor.")
        
        processor = self.processor_classes[spec_type](config)
        processor.start()


class Partition(Processor):

    def __init__(self, config):
        super().__init__(config)
        
        # a list of dictionaries with the unique combinations of values for the divisions
        self.info_by_division = []
        # self.info_by_division_by_layers is a list with these same unique values, repeated for
        # each unique layer
        self.info_dicts = self.info_by_division_by_layers if self.layers else self.info_by_division
 
        self.assign_data_sources()

    def start(self):
        self.process_divider(*next(iter(self.spec['divisions'].items())), self.spec['divisions'])

    def assign_data_sources(self):
       
        divider = self.spec['divisions'].get('data_source')
        if not divider:
            return
        # if data_source is 'all_animals' or similar, expand that into a list of identifiers
        if 'all' in divider.get('members', []):
            divider['members'] = [s.identifier for s in getattr(self.experiment, divider['members'])]
            
    def process_divider(self, divider_type, current_divider, divisions):

        info = {}
        
        for i, member in enumerate(current_divider['members']):

            info[divider_type] = member
            if divider_type == 'data_source':
                info['data_object_type'] = current_divider['type']
                
            self.advance_index(current_divider, i)
            
            updated_info = self.inherited_division_info | info

            if len(divisions) > 1:
                remaining_divisions = {k: v for k, v in divisions.items() if k != divider_type}
                self.process_divider(*next(iter(remaining_divisions.items())), remaining_divisions, 
                                     info=updated_info)
            else:
                is_final_value = i == len(current_divider['members']) - 1
                self.info_by_division.append(updated_info)
                wrap_up_args = {
                    'updated_info': updated_info, 'i': i, 'is_final_value': is_final_value
                }
                self.wrap_up(wrap_up_args)

    def advance_index(self, current_divider, i):
        if self.name == 'segment':
            return
        if 'dim' in current_divider:
            dim = current_divider['dim']
            self.current_index[dim] = self.starting_index[dim] + i

    def get_calcs(self):
        
        d = self.info_by_division[-1]

        for key in ['neuron_type', 'period_type', 'period_group']:
            if key in d:
                setattr(self, f"selected_{key}", d[key])

        data_source = self.get_data_sources(data_object_type = d['data_object_type'], 
                                            identifier=d['data_source'])
        
        attr = self.spec.get('attr', 'calc')

        if self.layers:
            self.get_layer_calcs()
        else:
            d.update({
                'attr': attr, 
                attr: getattr(data_source, attr), 
                data_source.name: data_source.identifier})
        

class Series(Partition):

    name = 'series'

    def __init__(self, config):
        super().__init__(config)
                
    def wrap_up(self, wrap_up_args):

        updated_info = wrap_up_args['updated_info']
        i = wrap_up_args['i']

        component = self.spec['components'][i]
        base = deepcopy(self.spec['base'])
        spec = recursive_update(base, component)

        self.start_next_processor(spec, updated_info)           


class Section(Partition):

    name = 'section'
    
    def __init__(self, config):
        super().__init__(config)

    def wrap_up(self, wrap_up_args):

        updated_info = wrap_up_args['updated_info']
  
        cell = self.child_layout.cells[*self.current_index]

        if not self.next:
            self.get_calcs()
            self.executive_plotter.delegate(
                cell, info=[self.info_dicts.pop()], spec=self.spec, is_last=not self.next)

        else:
            self.start_next_processor(self.next, updated_info)
          

class Segment(Partition):

    name = 'segment'

    def __init__(self, config):
        super().__init__(config)

    def wrap_up(self, wrap_up_args): 

        is_final_value = wrap_up_args['is_final_value']
        self.get_calcs()
        cell = self.parent_layout.cells[*self.current_index]

        if is_final_value:
            self.executive_plotter.delegate(
                cell, info=self.info_by_division, spec=self.spec, is_last=True)
            

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

