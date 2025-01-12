from copy import deepcopy
from k_onda.utils import recursive_update


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
    


