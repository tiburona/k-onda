from copy import deepcopy
from k_onda.utils import recursive_update
from ..plotting_helpers import smart_title_case


class ProcessorMixin:

    def init_spec_element(self, element):
        element_dict = deepcopy(self.spec.get(element, {}))
        if self.parent_processor:
            element_dict.update(getattr(self.parent_processor, element, {}))
        return element_dict


class LayerMixin(ProcessorMixin):

    def __init__(self):
        self.info_by_division_by_layers = []

    def init_layers(self):
        return self.init_spec_element('layer')
    
    def get_layer_calcs(self, d, data_source):
        for i, layer in enumerate(self.layers):

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
    

class AestheticsMixin(ProcessorMixin):
 
    def init_aesthetics(self):
        element_dict = deepcopy(self.spec.get('aesthetics', {}))
        if self.parent_processor:
            getattr(self.parent_processor, 'aesthetics', {}).update(element_dict)
        return getattr(self.parent_processor, 'aesthetics', {})
    

class LegendMixin:

    # TODO: why don't I make it so that this can work either way:  you can spec 
    # your colorbar as a legend around the subfigure, and in the init function
    # of the processor it adds to the list self.legend_info_list

    # or you can spec it as 'each' and wrap_up with the now updated division info
    # and the figure from the cell in the child layout

    @property
    def has_colorbar(self):
        return bool(self.colorbar_spec)

    @property
    def colorbar_spec(self):
        return self.spec.get('legend', {}).get('colorbar', {})
            
    @property
    def colorbar_for_each_plot(self):
        return self.has_colorbar and self.colorbar_spec.get('share') in ['each', None]

    @property
    def global_colorbar(self):
        return self.has_colorbar and self.colorbar_spec.get('share') == 'global'

    

class LabelMixin:

    def get_label(self):
       return self.spec.get('label', {})

    def get_next_label(self):
        if not self.next:
            return None
        else:
            next_spec = list(self.next.values())[0]
            return next_spec.get('label', {})
        
    def set_label(self, cell):
        if not self.label:
            return
        
        kwargs = {}
        
        for position in self.label:

            # get text of label
            text = self.fill_fields(self.label[position]['text'])
            if not self.label.get('smart_label', False):
                text = smart_title_case(text.replace('_', ' '))

            label_figure = self.child_layout.label_figure
            # get label_setter and kwargs
            kwargs = self.label[position].get('kwargs', {})
            if position == 'title':
                label_setter = getattr(label_figure, 'suptitle')
            else:
                if position in 'xy':
                    label_setter = getattr(label_figure, f'sup{position}label')   
                elif position in ['x_ax', 'y_ax']:
                    if position in 'y_ax':
                        print(f'y_ax {self.current_index}')
                    which = self.label[position].get('which', 'all')
                    if which and which != 'all':
                        if not cell.is_in_extreme_position(
                            position[0], 'last' in which, 'absolute' in which):
                            continue
                    label_setter = getattr(cell, f'set_{position[0]}label')
                else:
                    raise ValueError(f"Unknown label position: {position}")
                
            # set label
            label_setter(text, **kwargs)

    def is_condition_met(self, category, member, **_):
        
        return (
            getattr(self, f'selected_{category}', None) == member or
            self.selected_conditions.get(category) == member or
            category == 'period_types' and member in self.selected_period_types
            )


class MarginMixin:

    def calculate_rect(self, margin_spec):
        width = 1 - (margin_spec['right'] + margin_spec['left'])
        height = 1 - (margin_spec['top'] + margin_spec['bottom'])
        rect = (margin_spec['left'], margin_spec['bottom'], width, height)
        return rect
    
    def calculate_margins(self, margin_spec):
        margin_spec = deepcopy(margin_spec)
        for k, v in margin_spec.items():
            if k in ('top', 'right'):
                margin_spec[k] = 1 - v
        return margin_spec
            






            
                           


