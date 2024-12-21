from copy import deepcopy
import os
from neo.rawio import BlackrockRawIO
from matlab_interface import MatlabInterface


class PrepMethods:

    def construct_path(self, constructor_id):
        constructor = deepcopy(self.experiment.exp_info['path_constructors'][constructor_id])
        for field in constructor['fields']:
            constructor[field] = getattr(self, field)
        return constructor['template'].format(**constructor)
    
    def load_blackrock_file(self, file_path, nsx_to_load=None):
        reader = BlackrockRawIO(filename=file_path, nsx_to_load=nsx_to_load)
        reader.parse_header()
        return reader
    
  

