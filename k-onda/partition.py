from plotter_base import PlotterBase
from subplotter import Subplotter, BrokenAxes, AxWrapper

import numpy as np
from copy import deepcopy

class Partition(PlotterBase):

    def __init__(self, origin_plotter, parent_plotter=None, 
                 parent_processor = None, info=None):
        super().__init__()
        self.origin_plotter = origin_plotter
        self.parent_plotter = parent_plotter
        self.spec = self.active_spec
        self.next = None
        self.parent_processor = parent_processor

        for k in ('segment', 'section'):
            if k in self.spec:
                self.next = {k: self.spec[k]}
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
        self.info_by_attr = {}
        self.processor_classes = {
            'section': Section,
            'segment': Segment,
            'subset': Subset
        }

    def start(self):
        self.assign_data_sources()
        self.process_divider(*next(iter(self.spec['divisions'].items())), self.spec['divisions'])

    def assign_data_sources(self):
        for divider_type, divider in self.spec['divisions'].items():
            if divider_type == 'data_source' and 'all' in divider['members']:
                divider['members'] = [
                    source for source in getattr(self.experiment, divider['members']) 
                    if source.include()]
            

    def process_divider(self, divider_type, current_divider, divisions, info=None):
        
        info = info or {}
        
        for i, member in enumerate(current_divider['members']):
            
            if divider_type == 'data_type':
                info['data_source'] = source
                info[source.name] = source.identifier
               
            else:
                info[divider_type] = member
                
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
        for d in self.info_by_division:
            # Set selected attributes if applicable
            for key in ['neuron_type', 'period_type', 'period_group']:
                if key in d:
                    setattr(self, f"selected_{key}", d[key])
            
            # Determine the list of attributes, with a fallback if none are found
            attrs = [layer['attr'] for layer in self.layers if 'attr' in layer] or [
                self.active_spec.get('attr', 'calc')]
           
            d.update({attr: getattr(d['data_source'], attr) for attr in attrs})

class Section(Partition):
    def __init__(self, origin_plotter, parent_plotter=None,
                  index=None, parent_processor=None):
        super().__init__(origin_plotter, parent_plotter=parent_plotter, 
                         parent_processor=parent_processor)
        
        self.set_members()
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
            
    def set_members(self):
        data_source_spec = self.spec['divisions'].get('data_source', {})
        if type(data_source_spec.get('members')) in [int, str]:
            members = self.get_data_sources(
                **{k: data_source_spec[v] 
                   for k, v in zip(['identifier', 'data_object_type'], ['members', 'type'])})
            data_source_spec['members'] = members


    # @property 
    # def dimensions_of_subplot(self):
    #     dims = [1, 1]
    #     for division in self.spec.values():
    #         if 'dim' in division:
    #             length = len(division['members'])
    #             # if division['dim'] in (self.active_spec.get('break_axes') or {}):
    #             #     length *= len(self.active_spec['break_axes'][division['dim']])
    #             dims[division['dim']] = len(division['members'])
    #     return dims

    def set_dims(self, current_divider, i):
        if 'dim' in current_divider:
            dim = current_divider['dim']
            self.current_index[dim] = self.starting_index[dim] + i


    def wrap_up(self, *_):
        print("starting_index", self.starting_index)
        print("current_index", self.current_index)
        self.active_acks = self.active_plotter.axes[*self.current_index]
        self.active_plotter.apply_aesthetics(self.aesthetics)
        
        if self.next:
            # do next thing
            pass
        else:
            self.get_calcs()
            self.origin_plotter.delegate([self.info_by_division.pop()])


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
        self.get_calcs()
        if i == len(current_divider['members']) - 1:
            self.origin_plotter.delegate(self.info_by_division)


class Subset:
    pass


