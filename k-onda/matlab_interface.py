import numpy as np
import os
import uuid
import subprocess
import time


class MatlabInterface:

    def __init__(self, config):
        self.base_directory = f'/Users/katie/likhtik/data/temp'
        self.data_file_path = ''
        self.script_file_path = ''
        self.result_file_path = ''
        self.session_directory = ''

    def init_session(self):
        # Create a unique subdirectory for this session
        session_id = str(uuid.uuid4())
        self.session_directory = os.path.join(self.base_directory, session_id)
        os.makedirs(self.session_directory, exist_ok=True)

        # Create unique file paths within the session directory
        self.data_file_path = os.path.join(self.session_directory, 'temp_data.txt')
        self.script_file_path = os.path.join(self.session_directory, 'temp_script.m')

    def mtcsg(self, data, *args):
        str_args = ','.join([str(arg) for arg in args])
        execution_line = f"[yo, fo, to] = mtcsg(data, {str_args});\n"
        result = self.execute_function(data, execution_line, results=('yo', 'fo', 'to'))
        return result

    def execute_function(self, data, execution_line, results=('result')):
        self.init_session()
        np.savetxt(self.data_file_path, data)
        results_paths = [os.path.join(self.session_directory, result + '.txt') for result in results]
        with open(self.script_file_path, 'w') as script_file:
            script_file.write(f"addpath(genpath('/Users/katie/likhtik'));\n")
            script_file.write(f"data = load('{self.data_file_path}');\n")
            script_file.write(execution_line)
            for result in results:
                result_path = os.path.join(self.session_directory, result + '.txt')
                script_file.write(f"save('{result_path}', '{result}', '-ascii');\n")
        subprocess.run(["/Applications/MATLAB_R2022a.app/bin/matlab", "-batch", f"run('{self.script_file_path}')"])

        timeout = 120
        start_time = time.time()
        while not all([os.path.exists(path) for path in results_paths]):
            if time.time() - start_time >= timeout: #
                raise TimeoutError(f"Timeout after {timeout} seconds while waiting for file {self.result_file_path}")
            time.sleep(1)  # Wait for 1 second

        result = tuple(np.loadtxt(result_path) for result_path in results_paths)
        # shutil.rmtree(self.session_directory)

        return result









