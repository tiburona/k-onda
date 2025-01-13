from copy import deepcopy

from neo.rawio import BlackrockRawIO


class PrepMethods:
    
    def load_blackrock_file(self, file_path, nsx_to_load=None):
        reader = BlackrockRawIO(filename=file_path, nsx_to_load=nsx_to_load)
        reader.parse_header()
        return reader
    
  

