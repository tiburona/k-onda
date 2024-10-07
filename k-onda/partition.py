from plotter_base import PlotterBase
from subplotter import Subplotter

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
                self.next = {k: self.spec.pop(k)}
        self.gs_xy = self.spec.pop('gs_xy') if 'gs_xy' in self.spec else None

        if self.active_fig == None:
            self.fig = self.origin_plotter.make_fig()
            self.parent_plotter = self.active_plotter
        else:
            self.fig = self.active_fig
        self.inherited_info = info if info else {}
        self.info_list = []
        self.processor_classes = {
            'section': Section,
            'segment': Segment,
            'subset': Subset
        }

    def start(self):
        self.process_divider(*next(iter(self.spec.items())), self.spec)

    def process_divider(self, divider_type, current_divider, divisions):

        for i, member in enumerate(current_divider['members']):
            info = {}
            self.set_dims(current_divider, i)
            if divider_type in ['data_source', 'group', 'animal', 'unit', 'period', 'event']:
                if type(member) in [int, str]:
                    source = self.get_data_sources(
                        identifiers=[member], data_object_type=current_divider['type'])[0]
                else:
                    source = member
        
                info['attr'] = current_divider.get('attr', 'calc')
                info['data_source'] = source
                info[source.name] = source.identifier
            else:
                #setattr(self, f"selected_{divider_type}", member)
                info[divider_type] = member

            if len(divisions) > 1:
                remaining_divisions = {k: v for k, v in divisions.items() if k != divider_type}
                self.process_divider(*next(iter(remaining_divisions.items())), remaining_divisions)
            else:
                updated_info = self.inherited_info | info
                self.info_list.append(updated_info)
                self.wrap_up(current_divider, i)

                if self.next:
                    self.active_spec_type, self.active_spec = list(self.next.items())[0]
                    processor = self.processor_classes[self.active_spec_type](
                        self.origin_plotter, self.active_plotter, info=updated_info)
                    processor.start()

    def get_calcs(self):
        for d in self.info_list:
            for k, v in d.items():
                if k in ['neuron_type', 'period_type', 'period_group']:
                    setattr(self, f"selected_{k}", v) 
            attr = d['attr']
            d[attr] = getattr(d['data_source'], attr)
                 

class Section(Partition):
    def __init__(self, origin_plotter, parent_plotter=None,
                  index=None):
        super().__init__(origin_plotter, parent_plotter=parent_plotter)
        # index should refer to a starting point in the parent gridspec
        if index:
            self.starting_index = index
        elif self.gs_xy:
            self.starting_index = [dim[0] for dim in self.gs_xy]
        else:
            self.starting_index = [0, 0]
        self.current_index = deepcopy(self.starting_index)
    


        if index:
            a = 'foo'
           
        if 'aspect' in self.spec:
            self.aspect = self.spec.pop('aspect')
        else:
            self.aspect = None
        # right now section is assuming that it's taking one spot in a gridspec.
        # but in the layout case it needs to take

        # this thinks it needs to create a subplot, but in the layout case it already exists
        # can I make this either receive a subplot as an argument or create one?
        # how would I pass it down?

        if not self.is_layout:
            self.active_plotter = Subplotter(
                self.active_plotter, self.index, self.spec, aspect=self.aspect)


    @property 
    def dimensions_of_subplot(self):
        dims = [1, 1]
        for division in self.spec.values():
            if 'dim' in division:
                dims[division['dim']] = len(division['members'])
        return dims

    def set_dims(self, current_divider, i):
        if 'dim' in current_divider:
            dim = current_divider['dim']
            self.current_index[dim] = self.starting_index[dim] + i


    def wrap_up(self, *_):
        print("starting_index", self.starting_index)
        print("current_index", self.current_index)
        self.active_ax = self.active_plotter.axes[*self.current_index]

        if self.next:
            # do next thing
            pass
        else:
            self.get_calcs()
            self.origin_plotter.process_calc(self.info_list)


class Segment(Partition):
    def __init__(self, origin_plotter, parent_plotter, sources=None, info=None):
          super().__init__(origin_plotter, parent_plotter, sources, info=info)
          self.data = []
          self.columns = []

    def prep(self):
        pass
    
    def set_dims(self, *_):
        pass

    def wrap_up(self, current_divider, i): 
        self.get_calcs()
        if i == len(current_divider['members']) - 1:
            self.origin_plotter.process_calc(self.info_list)


class Subset:
    pass


