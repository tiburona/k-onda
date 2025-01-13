from copy import deepcopy
from k_onda.utils import recursive_update
from .plotting_helpers import smart_title_case


class ProcessorMixin:

    def init_spec_element(self, element):
        element_dict = deepcopy(self.spec.get(element, {}))
        if self.parent_processor:
            element.update(getattr(self.parent_processor, element))
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
    

class BorderMixin(ProcessorMixin):
    def init_border(self):
        return self.init_spec_element('border')
    

class LabelMixin:

    def label(self):
        for position in self.spec.get('label', []):
            if position in 'xy':
                label_setter = getattr(self.figure, f'sup{position}label')
                lab = self.spec['label'][position]    
            elif position == 'title':
                lab = self.fill_fields(self.spec['label'][position])
                label_setter = getattr(self.figure, 'suptitle')
            else:
                raise ValueError(f"Unknown label position: {position}")
                
            if not self.spec['label'].get('smart_label', False):
                lab = smart_title_case(lab.replace('_', ' '))

            label_setter(lab)






            
                           


