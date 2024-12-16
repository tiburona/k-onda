from plotting.plotter_base import PlotterBase
from plotting.subplotter import Subplotter
from utils.utils import recursive_update

from copy import deepcopy
from functools import reduce
import operator

class Partition(PlotterBase):

    def __init__(self, origin_plotter, parent_plotter=None, 
                 parent_processor = None, info=None):
        super().__init__()
        self.origin_plotter = origin_plotter
        self.parent_plotter = parent_plotter
        self.spec = self.active_spec
        self.next = None
        for k in ('segment', 'section'):
            if k in self.spec:
                self.next = {k: self.spec[k]}
        self.parent_processor = parent_processor
        self.assign_data_sources()
        self.total_calls = reduce(
            operator.mul, [len(div['members']) for div in self.spec['divisions'].values()], 1)
        self.remaining_calls = self.total_calls
        self.layers = self.spec.get('layers', {})
        if self.parent_processor:
            self.layers.update(self.parent_processor.layers)
        self.aesthetics = self.spec.get('aesthetics', {})
        if self.parent_processor:
            self.aesthetics.update(self.parent_processor.aesthetics)

        if self.active_fig == None:
            self.fig = self.origin_plotter.make_fig()
            self.parent_plotter = self.active_plotter
        else:
            self.fig = self.active_fig
        self.inherited_info = info if info else {}
        self.info_by_division = []
        self.info_by_division_by_layers = []
        self.info_dicts = self.info_by_division_by_layers if self.layers else self.info_by_division
        self.info_by_attr = {}
        self.processor_classes = {
            'section': Section,
            'segment': Segment,
            'subset': Subset
        }
        
    @property
    def last(self):
        return not self.remaining_calls and not self.next

    def start(self):
        self.process_divider(*next(iter(self.spec['divisions'].items())), self.spec['divisions'])

    def assign_data_sources(self):
        divider = self.spec['divisions'].get('data_source')
        if not divider:
            return
        if 'all' in divider.get('members', []):
            divider['members'] = [s.identifier for s in getattr(self.experiment, divider['members'])]
            
    def process_divider(self, divider_type, current_divider, divisions, info=None):

        info = info or {}
        
        for i, member in enumerate(current_divider['members']):

            info[divider_type] = member
            if divider_type == 'data_source':
                info['data_object_type'] = current_divider['type']
                
            self.set_dims(current_divider, i)
            
            updated_info = self.inherited_info | info

            if len(divisions) > 1:
                remaining_divisions = {k: v for k, v in divisions.items() if k != divider_type}
                self.process_divider(*next(iter(remaining_divisions.items())), remaining_divisions, 
                                     info=updated_info)
            else:
                
                self.info_by_division.append(updated_info)
                self.wrap_up(current_divider, i)

                if self.next:
                    self.active_spec_type, self.active_spec = list(self.next.items())[0]
                    processor = self.processor_classes[self.active_spec_type](
                        self.origin_plotter, self.active_plotter, info=updated_info)
                    processor.start()

    def get_calcs(self):
        
        d = self.info_by_division[-1]

        for key in ['neuron_type', 'period_type', 'period_group']:
            if key in d:
                setattr(self, f"selected_{key}", d[key])

        data_source = self.get_data_sources(data_object_type = d['data_object_type'], 
                                            identifier=d['data_source'])
        
        if self.layers:
            for i, layer in enumerate(self.layers):
                if i == 2:
                    a = 'foo'
                attr = layer.get('attr', self.active_spec.get('attr', 'calc'))
                if 'calc_opts' in layer:
                    recursive_update(self.calc_opts, layer['calc_opts'])
                    self.calc_opts = self.calc_opts
                new_d = deepcopy(d)
                new_d.update({
                    'layer': i, 
                    'attr': attr,
                    attr: getattr(data_source, attr), 
                    data_source.name: data_source.identifier})
                self.info_by_division_by_layers.append(new_d)
                
        else:
            attr = self.active_spec.get('attr', 'calc')
            d.update({
                    'attr': attr, 
                    attr: getattr(data_source, attr), 
                    data_source.name: data_source.identifier})
        

class Section(Partition):
    def __init__(self, origin_plotter, parent_plotter=None,
                  index=None, parent_processor=None):
        super().__init__(origin_plotter, parent_plotter=parent_plotter, 
                         parent_processor=parent_processor)
        
        # index should refer to a starting point in the parent gridspec
        self.gs_xy = self.spec.pop('gs_xy', None) 
        if index:
            self.starting_index = index
        elif self.gs_xy:
            self.starting_index = [dim[0] for dim in self.gs_xy]
        else:
            self.starting_index = [0, 0]
        self.current_index = deepcopy(self.starting_index)

        self.aspect = self.aesthetics.get('aspect')

        if not self.is_layout:
            self.active_plotter = Subplotter(
                self.active_plotter, self.current_index, self.spec, aspect=self.aspect)

    def set_dims(self, current_divider, i):
        if 'dim' in current_divider:
            dim = current_divider['dim']
            self.current_index[dim] = self.starting_index[dim] + i

    def wrap_up(self, current_divider, i):
        self.remaining_calls -= 1
        self.active_acks = self.active_plotter.axes[*self.current_index]
        self.active_plotter.apply_aesthetics(self.aesthetics)
        self.origin_plotter.label(self.info_by_division[-1], self.active_acks, self.aesthetics, 
                                  self.remaining_calls)

        if not self.next:
            self.get_calcs()
            self.origin_plotter.delegate([self.info_dicts.pop()], is_last=self.last)


class Segment(Partition):
    def __init__(self, origin_plotter, parent_plotter, info=None,
                 parent_processor=None):
          super().__init__(origin_plotter, parent_plotter, info=info,
                           parent_processor=parent_processor)
          self.data = []
          self.columns = []

    def prep(self):
        pass
    
    def set_dims(self, *_):
        pass

    def wrap_up(self, current_divider, i): 
        self.remaining_calls -= 1
        self.get_calcs()
        if self.last:
            self.origin_plotter.delegate(self.info_dicts, is_last=self.last)
            

class Subset:
    pass


