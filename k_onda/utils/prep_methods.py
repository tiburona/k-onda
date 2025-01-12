from copy import deepcopy

from neo.rawio import BlackrockRawIO


class PrepMethods:

    def construct_path(self, constructor_id):
        constructor = deepcopy(self.experiment.exp_info['path_constructors'][constructor_id])
        return self.fill_fields(constructor)
    
    def fill_fields(self, constructor):
        if not constructor:
            return
        if isinstance(constructor, str):
            return constructor
        
        for field in constructor['fields']:
            if field in self.selectable_variables:
                constructor[field] = getattr(self, field, getattr(self, f'selected_{field}'))
            else:
                constructor[field] = getattr(self, field)
        return constructor['template'].format(**constructor)
    
    def load_blackrock_file(self, file_path, nsx_to_load=None):
        reader = BlackrockRawIO(filename=file_path, nsx_to_load=nsx_to_load)
        reader.parse_header()
        return reader
    
  

