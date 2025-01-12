import numpy as np
import os
import uuid
import subprocess
import time
import scipy.io

from k_onda.utils import is_truthy


class MatlabInterface:

    def __init__(self, config, tags=None):
        self.path_to_matlab = config['path_to_matlab']
        self.paths_to_add = config.get('paths_to_add', [])
        self.recursive_paths_to_add = config.get('recursive_paths_to_add', [])
        self.base_directory = config['base_directory']
        self.data_file_path = ''
        self.script_file_path = ''
        self.result_file_path = ''
        self.session_directory = ''
        self.tags = tags

    def init_session(self):
        # Create a unique subdirectory for this session
        session_id = str(uuid.uuid4())
        if self.tags:
            session_id += '_'.join([str(t) for t in self.tags])
        self.session_directory = os.path.join(self.base_directory, session_id)
        os.makedirs(self.session_directory, exist_ok=True)

        # Create unique file paths within the session directory
        self.data_file_path = os.path.join(self.session_directory, 'temp_data.txt')
        self.script_file_path = os.path.join(self.session_directory, 'temp_script.m')

    def mtcsg(self, data, *args):
        str_args = ','.join([str(arg) for arg in args])
        execution_line = f"[yo, fo, to] = mtcsg(data, {str_args});\n"
        result = self.execute_function(data, execution_line, results=['yo', 'fo', 'to'])
        return result
    
    def open_nsx(self, file_path):
        file_dir = os.path.dirname(file_path)
        file_path += '.ns3'

        execution_line = f"""
            openNSx('{file_path}');\n
            data = NS3.Data;\n
            h5create('{file_dir}/output_data.h5', '/NS3_Data', size(data), 'Datatype', 'int16');\n
            h5write('{file_dir}/output_data.h5', '/NS3_Data', data);\n
        """
        self.execute_function('', execution_line, input_type='string')
        return os.path.join(file_dir, 'output_data.h5')

    def filter(self, data):
        execution_line = f"filtered_data = removeLineNoise_SpectrumEstimation(data', 2000, ['NH=5','LF=60']);"
        result = self.execute_function(data, execution_line, results=['filtered_data'])
        return result
    
    def tsdata_to_info_crit(self, data, momax=200, icregmode="''"):
        execution_line = f"[AIC,BIC,moAIC,moBIC] = tsdata_to_infocrit(data, {momax}, {icregmode});"
        result = self.execute_function(data, execution_line, results=['moAIC', 'moBIC'])
        return result
    
    def granger_causality(self, data, momax=200, icregmode="''"):
        execution_line = f"""
            [AIC,BIC,moAIC,moBIC] = tsdata_to_infocrit(data, {momax}, {icregmode});
            morder = moBIC;
            [A,SIG] = tsdata_to_var(data,morder,{icregmode});
            [G,info] = var_to_autocov(A,SIG,[]);
            f = autocov_to_spwcgc(G,1000);
            assert(~isbad(f,false),'spectral GC calculation failed');
        """
        result = self.execute_function(data, execution_line, results=['f'], ext='mat')
        return result

    
    def execute_function(self, data, execution_line, results=[], ext='txt', input_type='numeric'):
        self.init_session()
        if input_type == 'numeric':
            np.savetxt(self.data_file_path, data)
        else:
            with open(self.data_file_path, 'w') as f: f.write(data)

        results_paths = [os.path.join(self.session_directory, result + f'.{ext}') for result in results]
        with open(self.script_file_path, 'w') as script_file:
            for p in self.recursive_paths_to_add:
                script_file.write(f"addpath(genpath('{p}'));\n")
            for p in self.paths_to_add:
                script_file.write(f"addpath('{p}');\n")
            if is_truthy(data):
                script_file.write(f"data = load('{self.data_file_path}');\n")
            script_file.write(execution_line)
            for result in results:
                result_path = os.path.join(self.session_directory, result + f'.{ext}')
                ascii = ", '-ascii'" if ext == 'txt' else ""
                save_line = f"\nsave('{result_path}', '{result}'{ascii});\n"
                script_file.write(save_line)
        subprocess.run([self.path_to_matlab, "-batch", f"run('{self.script_file_path}')"])

        timeout = 360
        start_time = time.time()
        while not all(os.path.exists(path) for path in results_paths):
            if time.time() - start_time >= timeout:
                raise TimeoutError(f"Timeout after {timeout} seconds while waiting for file {self.result_file_path}")
            time.sleep(1)  # Wait for 1 second
        if ext == 'txt':
            load_fun = np.loadtxt
        else:
            load_fun = scipy.io.loadmat
        result = tuple(load_fun(result_path) for result_path in results_paths)
        # shutil.rmtree(self.session_directory)

        return result









