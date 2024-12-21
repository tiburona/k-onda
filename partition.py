from plotter_base import PlotterBase
from subplotter import CellArray, BrokenAxes, AxWrapper

import numpy as np
from copy import deepcopy
from functools import reduce
import operator


class PlotProcessor(PlotterBase):
    
    def __init__(self, origin_plotter, parent_cell_array=None, 
                 parent_processor = None, info=None):
        self.origin_plotter = origin_plotter
        self.parent_plotter = parent_cell_array
        self.spec = self.active_spec
        self.parent_processor = parent_processor
        self.aesthetics = self.spec.get('aesthetics', {})
        if self.parent_processor:
            self.aesthetics.update(self.parent_processor.aesthetics)
        if 'calc_spec' in self.spec:
            self.calc_spec = self.spec['calc_spec']
            
        if self.active_fig == None:
            self.fig = self.origin_plotter.make_fig()
            self.parent_cell_array = self.active_cell_array
        else:
            self.fig = self.active_fig
        
        self.processor_classes = {
            'split': Splitter,
            'segment': Segmenter,
            'subset': Subsetter,
            'section': Splitter, 
            'arrange': Arranger
        }


class Partitioner(PlotProcessor):

    def __init__(self, origin_plotter, parent_plotter=None, 
                 parent_processor = None, info=None):
        super().__init__(origin_plotter, 
                         parent_plotter, 
                         parent_processor=parent_processor,
                         info=info)
        
        self.next = None
        for k in ('segment', 'split'):
            if k in self.spec:
                self.next = {k: self.spec[k]}
        self.assign_data_sources()
        self.total_calls = reduce(
            operator.mul, [len(div['members']) for div in self.spec['divisions'].values()], 1)
        self.remaining_calls = self.total_calls
        self.layers = self.spec.get('layers', {})
        if self.parent_processor:
            self.layers.update(self.parent_processor.layers)
        self.inherited_info = info if info else {}
        self.info_by_division = []
        self.info_by_attr = {}       
        
    @property
    def last(self):
        return not self.remaining_calls and not self.next

    def start(self):
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
            
            if divider_type == 'data_source':
                info['data_source'] = member
                info[member.name] = member.identifier
               
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
                        self.origin_plotter, self.active_cell_array, info=updated_info)
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
            

class Sectioner(Partitioner):
    
    def __init__(self, origin_plotter, parent_plotter=None, 
                 parent_processor = None, info=None, index=None):
        super().__init__(origin_plotter, parent_plotter, parent_processor, info)
        self.components = self.spec['arrange']['members']
        self.gs_args = self.spec['gs_args']
        
        self.active_cell_array = CellArray(
            self.active_cell_array, self.current_index, self.spec, aspect=self.aspect)
        
    def wrap_up(self, current_divider, i):
        print("starting_index", self.starting_index)
        print("current_index", self.current_index)
        self.remaining_calls -= 1
        self.active_cell_array.apply_aesthetics(self.aesthetics)
        
        for i, row in enumerate(self.components):
            for j, component in row:
                self.origin_plotter.plot(component['calc_spec'], component['graph_opts'], 
                               parent=self.active_cell_array)
                 
                

class Splitter(Partitioner):
    def __init__(self, origin_plotter, parent_plotter=None,
                  index=None, parent_processor=None):
        super().__init__(origin_plotter, parent_plotter=parent_plotter, 
                         parent_processor=parent_processor, index=index)
         
        self.starting_index = index if index else [0, 0]
        self.current_index = deepcopy(self.starting_index)

        self.aspect = self.aesthetics.get('aspect')
        
        self.active_cell_array = CellArray(
            self.active_cell_array, self.current_index, self.spec, aspect=self.aspect)
            

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


    def wrap_up(self, current_divider, i):
        print("starting_index", self.starting_index)
        print("current_index", self.current_index)
        self.remaining_calls -= 1
        self.active_cell = self.active_plotter.axes[*self.current_index]
        self.active_cell_array.apply_aesthetics(self.aesthetics)
        
        if self.next:
            # do next thing
            pass
        else:
            self.get_calcs()
            if self.last:
                print("self.last!")
            self.origin_plotter.delegate([self.info_by_division.pop()], is_last=self.last)


class Segmenter(Partitioner):
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
        if i == len(current_divider['members']) - 1:
            self.origin_plotter.delegate(self.info_by_division, is_last=self.last)
            


class Subsetter:
    pass


class Arranger(PlotProcessor):
    
    def __init__(self, origin_plotter, parent_plotter=None, 
                 parent_processor = None, info=None, index=None):
        super().__init__(origin_plotter, 
                         parent_plotter, 
                         parent_processor=parent_processor,
                         info=info)
        self.members = self.spec['members']
        self.starting_index = index if index else [0, 0]
        self.active_cell_array = CellArray(self.active_cell_array, self.starting_index, 
                                           spec=self.active_spec)
        
    def start(self):
        for member in self.members:
            self.origin_plotter.process_plot_spec(member['plot_spec'], 
                                                  index=member['index'])





