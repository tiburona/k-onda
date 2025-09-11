from copy import copy, deepcopy

from .processor import Processor
from k_onda.utils import recursive_update


class Partition(Processor):
    """
    A Processor that divides the data into parts. It recurses through a list of `divisions`, each of 
    which is a dictionary with a `divider_type` and a list of `members`. Once the partition has 
    defined a unique combinations of members, it calls `wrap_up` to get the calcs for the unique 
    combination, or to start the next processor(s) if the partition is not the final.
    """

    def __init__(self, config):
        super().__init__(config)

    def setup_unique(self):
        # self.info_by_division_by_layers is a list with these same unique values, repeated for
        # each unique layer
        self.info_dicts = self.info_by_division_by_layers if self.layers else self.info_by_division
        self.assign_data_sources()

    def finalize_init_unique(self):
        if self.global_colorbar:
            self.legend_info_list.append((self.figure, self.colorbar_spec, []))

    def start(self, top_level=False):
        if self.spec.get('default_period_type'):
            self.selected_period_type = self.spec['default_period_type']
        self.process_divisions(self.spec['divisions'])
        if top_level:
            if self.layers:
                for i, layer in enumerate(self.layers):
                    self.executive_plotter.delegate(
                        info=self.info_by_division_by_layers[i], 
                        spec=recursive_update(deepcopy(self.spec), layer), 
                        aesthetics=recursive_update(
                            deepcopy(self.aesthetics), layer.get('aesthetics', {})),
                        plot_type=layer.get('plot_type', self.plot_type)
                    )
            else:
                self.executive_plotter.delegate(
                    info=self.info_by_division, spec=self.spec, spec_type=self.name, 
                    plot_type=self.plot_type, aesthetics=self.aesthetics, 
                    legend_info_list=self.legend_info_list)

    def assign_data_sources(self):
        """
        Assign data sources to each division. If a division's 'members' key is a string beginning 
        with 'all', fetches the identifiers for the relevant data objects and assigns them to 
        'members'. E.g., 'all_animals' assigns a list of the identifiers of all the animals in the 
         experiment."""

        for division in self.spec['divisions']:
            data_source = division.get('data_source')
            
            if not data_source:
                return
            # if data_source is 'all_animals' or similar, expand that into a list of identifiers
            if 'all' in division.get('members', []):
                division['members'] = [
                    s.identifier for s in getattr(self.experiment, division['members'])]
                
    def copy_info(self, info):
        # can't deepcopy info with cell in it; causes infinite loop 
        cell = info.pop('cell', None)
        new_info = deepcopy(info) 
        info['cell'] = cell
        new_info['cell'] = cell
        return new_info
    
    def update_info(self, info, member, divider_type):
         # Merge it into our accumulated info
        if isinstance(member, dict):
            # member is dict, as in conditions, period_types, etc.
            updated_info = {**info, divider_type: {**info.get(divider_type, {}), **member}}
        else:
            updated_info = {**info, divider_type: member}
        return updated_info

            
    def process_divisions(self, divisions, info=None):
        """
        Recursively process a list of divider dicts, building up a cartesian product.
        Each divider dict looks like:
            {
                'divider_type': 'conditions',
                'members': [...],
                'dim': ...,
            }
       
        """
        
        if not info:
            info = self.copy_info(self.inherited_division_info)   

        # we hit a leaf in the recursion
        if not divisions:
            # This is the final combination of all previous divider choices.        
            self.wrap_up(info)
            return

        # Otherwise, take the first divider in the list
        divider = divisions[0]
        divider_type = divider['divider_type']
        info['data_source'] = divider.get('data_source', info.get('data_source', self.inherited_division_info.get('data_source')))

        # Go through each of its members
        for i, member in enumerate(divider['members']):
            # update the info dict with information about each member
            updated_info = self.update_info(info, member, divider_type)
            # advance the index to indicate the position of this member
            self.advance_index(divider, i)
            # Now recurse on the remainder of the list, carrying `updated_info`
            self.process_divisions(divisions[1:], info=updated_info)

    def advance_index(self, current_divider, i):
        if 'dim' in current_divider:
            dim = current_divider['dim']
            self.current_index[dim] = self.starting_index[dim] + i

    def wrap_up(self, updated_info): 
         
        # set vals before you label because labels often use vals, e.g. labeling the period_type
        self.set_vals(updated_info)
        cell = self.child_layout.cells[*self.current_index]
        # labels are applied at every level so they go on the appropriate subfigure
        # TODO: something here needs to be tracking the index in the larger figure 
       
        self.set_label(cell, updated_info)

        if self.colorbar_for_each_plot:
            self.legend_info_list.append((cell, self.colorbar_spec, [updated_info]))

        if not self.next:
            updated_info['cell'] = cell
            updated_info['index'] = copy(self.current_index) 
            updated_info['last_spec'] = self.spec
                       
            if self.layers:
                self.get_layer_dicts(updated_info)
            else:
                self.info_by_division.append(updated_info)
                self.get_calcs(updated_info)
       
           

        else:
            self.start_next_processor(self.next, updated_info, self.info_by_division, self.info_by_division_by_layers)

    def set_vals(self, info):
        # TODO: Make sure you've thoroughly thought through *unsetting* attributes 
        # for later plots in which those attributes aren't divisions.
        # Maybe add an attribute to Base that's a list called already_set_attributes
        # and also keep a running list of division_types in a processor, and if when
        # you start a processor if you see that something is set that's nowhere in your 
        # running list of already encountered divisions, you unset it.
        for key in ['neuron_type', 'period_type', 'period_group', 'period_types', 'conditions']:
            if key in info:     
                setattr(self, f"selected_{key}", info[key])


    def get_calcs(self, info):

        if not info.get('data_source'):
            data_source = self.experiment
        else:
            data_source = self.get_data_sources(data_object_type = info['data_source'], 
                                            identifier=info[info['data_source']])
        
        attr = info.get('attr', 'calc')

        info.update({
            'attr': attr, 
            attr: getattr(data_source, attr), 
            data_source.name: data_source.identifier})
        
        if self.legend_info_list:
            self.legend_info_list[-1][2].append(info)
        

class Series(Partition):
    """
    A processor that allows plotting several `components` for a unique combination of divisions in a 
    partition. For example, if you were to plot both a waveform and a PSTH for any of several 
    neurons, you would use a Series processor. The waveform and the PSTH would be the `components`, 
    and the neurons would be the `divisions`.
    """

    name = 'series'

    def __init__(self, config):
        super().__init__(config)
                
    def wrap_up(self, updated_info):
        
        self.label(cell=None)
        for component in self.spec['components']:
            base = deepcopy(self.spec['base'])
            spec = recursive_update(base, component)
            self.start_next_processor(spec, updated_info, self.info_by_division) 
                

class Section(Partition):

    name = 'section'
  
    
    def __init__(self, config):
        super().__init__(config)
        

class Segment(Partition):

    name = 'segment'

    def __init__(self, config):
        super().__init__(config)
    
    def get_layout_args(self):
        """
        Called by a Processor when creating a child layout; establishes that a Segment's child 
        layout is always a 1x1 grid containing a single ax.
        """
        return {'dimensions': [1, 1]}  
    
    def advance_index(self, *_):
        # A Segment does not advance its index, as it only contains a single ax.

        return 
           

class Split(Partition):
    name = 'split'

    def __init__(self, config):
        super().__init__(config)
        self.splits = True

    # a split is unlike other partitions in that it needs to expand the number of 
    # entries in info, while keeping the split label on each entry.  
    # within each split, the divider type and members should be labeled

    # in this sense, a split is like layers.  The other partitions need to be split-aware
    # and if there is a split, understand that they must iterate through splits.
    # if there are splits, each subsequent operation must operate on each split, 
    # and each info by division, or info by division by list

    # update_info was the method that did this before.  it now must iterate through the splits
    # 

    # if divider_type is a data source than there are three options:
    # all_x, a list of identifiers, or a list of lists (actually I think just two.  I think assign_data_sources still works)

    # if divider_type is a selectable variable than there are two options:
    # a list of identifiers, or a list of lists.  

    # this class can overwrite wrap_up so the only thing it does is start the next processor

    

    def process_divisions(self, divisions, info=None):
        
        
        if not info:
            info = self.copy_info(self.inherited_division_info)   

        # we hit a leaf in the recursion
        if not divisions:
            # This is the final combination of all previous divider choices.        
            self.wrap_up(info)
            return

        # Otherwise, take the first divider in the list
        divider = divisions[0]
        divider_type = divider['divider_type']
        info['data_source'] = divider.get('data_source', self.inherited_division_info.get('data_source'))

        # Go through each of its members
        for i, member in enumerate(divider['members']):
            # update the info dict with information about each member
            updated_info = self.update_info(info, member, divider_type, i)
            # advance the index to indicate the position of this member
            
            # Now recurse on the remainder of the list, carrying `updated_info`
            self.process_divisions(divisions[1:], info=updated_info)

    def update_info(self, info, member, divider_type, index):
         # Merge it into our accumulated info
        if isinstance(member, dict):
            # member is dict, as in conditions, period_types, etc.
            updated_info = {**info, divider_type: {**info.get(divider_type, {}), **member}, 
                            'split_index': index}
        else:
            updated_info = {**info, divider_type: member, 'split_index': index}

        return updated_info
    
    def wrap_up(self, info):
                
        if self.next:
            self.start_next_processor(self.next, info, self.info_by_division, self.info_by_division_by_layers)

            