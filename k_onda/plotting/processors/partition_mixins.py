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
        return self.init_spec_element('aesthetics')
    

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

    def label(self, cell=None):
        label = self.construct_spec_based_on_conditions(self.spec.get('label', {}))
        
        label_pad = label.pop('label_pad', 0)
        for position in label:

            # get text of label
            text = self.fill_fields(label[position])
            if not self.spec['label'].get('smart_label', False):
                text = smart_title_case(text.replace('_', ' '))

            # get label_setter and kwargs
            if position == 'title':
                label_setter = getattr(self.figure, 'suptitle')
                kwargs = {}
            else:
                kwargs = {'y' if position[0] == 'x' else 'x': label_pad} if label_pad else {}

                if position in 'xy':
                    label_setter = getattr(self.figure, f'sup{position}label')   
                elif position in ['x_ax', 'y_ax']:
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
            






            
                           


