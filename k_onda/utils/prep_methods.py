from copy import deepcopy

from neo.rawio import BlackrockRawIO

import os
from k_onda.interfaces import MatlabInterface 


class PrepMethods:
    
    def load_blackrock_file(self, file_path, nsx_to_load=None):
        try:
            reader = BlackrockRawIO(filename=file_path, nsx_to_load=nsx_to_load)
            reader.parse_header()
        except KeyError as e:
            print(f"KeyError: {e}")
            file_dir = os.path.dirname(file_path)
            ml = MatlabInterface(self.env_config['matlab_config'])
            if not os.path.exists(os.path.join(file_dir, 'output_data.h5')):
                ml.open_nsx(file_dir)
        return reader
    