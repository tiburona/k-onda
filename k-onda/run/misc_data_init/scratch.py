from copy import deepcopy
import os
from neo.rawio import BlackrockRawIO



'/Users/katie/likhtik/CH27mice/CH272/CH272_HABCTXB'


def load_blackrock_file(file_path, nsx_to_load=None):
    reader = BlackrockRawIO(filename=file_path, nsx_to_load=nsx_to_load)
    reader.parse_header()
    return reader

reader = load_blackrock_file('/Users/katie/likhtik/CH27mice/CH272/CH272_HABCTXB')

a = 'foo'