import os, uuid
from .base import Base
from k_onda.utils import smart_title_case, find_container_with_key

class OutputGenerator(Base):

    def __init__(self):
        self.file_path = ''
        self.opts_file_path = ''

    def build_write_path(self, file_type='plot', opts=None):
            
        constructors = self.io_opts['paths']
        self.write_opts = constructors.get(file_type, constructors.get('out', '.'))
        default_ext = 'png' if file_type == 'plot' else 'csv'

        if isinstance(self.write_opts, str):

            if '{'  in self.write_opts:
                data_source_dict = find_container_with_key(opts, 'data_source')

                if 'identifier' in self.write_opts:
                    self.write_opts = self.fill_fields(
                        self.write_opts, 
                        identifier='_'.join(data_source_dict['members']),
                        data_source_dict=data_source_dict)
                else:
                    self.write_opts = self.fill_fields(self.write_opts)
            
            if '.' in self.write_opts:
                    self.file_path = self.write_opts
            else:
                self.file_path = self.write_opts + f'.{default_ext}'

            self.opts_file_path = self.file_path[0:-3] + 'txt'
            return
        
        else: # is dict
            
            root, fname, path = [
                self.fill_fields(self.write_opts.get(key)) 
                for key in ['root', 'fname', 'path']
            ]
            
            # If user explicitly set 'path', we skip building our own path
            if path:
                self.file_path = path
                self.opts_file_path = self.file_path[0:-3] + 'txt'
                return
        
            else:
                # Fallback to some default root
                if not root:
                    root = os.getcwd()
                
                # Fallback to some default fname
                if not fname:
                    fname = '_'.join([self.kind_of_data, self.calc_type])
                
                self.title = smart_title_case(fname.replace('_', ' '))
                
                # Build partial path (no extension yet)
                self.file_path = os.path.join(root, self.kind_of_data, fname)

                self.handle_collisions()
            
                # Build final file path and an opts file
                self.opts_file_path = self.file_path + '.txt'

                default_ext = 'png' if file_type == 'plot' else 'csv'

                if isinstance(self.write_opts, dict):
                    ext = self.write_opts.get('extension', default_ext)
                else:
                    ext = '.png'
                self.file_path += ext

    def handle_collisions(self):
        if os.path.exists(self.file_path) and not self.write_opts.get('allow_overwrite', True):
            uuid_str = str(uuid.uuid4())[:self.write_opts.get('unique_hash', 8)]
            basename = os.path.basename(self.file_path)
            dir_ = os.path.dirname(self.file_path)
            new_name = f"{basename}_{uuid_str}"
            self.file_path = os.path.join(dir_, new_name)
