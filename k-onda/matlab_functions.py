import numpy as np
import os
import subprocess
import uuid
import shutil



def xcorr(data, lags):
    base_directory = '/Users/katie/likhtik/data/temp'

    # Generate a unique id for this session
    session_id = str(uuid.uuid4())

    # Create a unique subdirectory for this session
    session_directory = os.path.join(base_directory, session_id)
    os.makedirs(session_directory, exist_ok=True)

    # Create unique file paths within the session directory
    data_file_path = os.path.join(session_directory, 'temp_data.txt')
    script_file_path = os.path.join(session_directory, 'temp_script.m')
    result_file_path = os.path.join(session_directory, 'temp_result.txt')

    # Save data to temporary file
    np.savetxt(data_file_path, data)

    # Create MATLAB script
    with open(script_file_path, 'w') as script_file:
        script_file.write(f"data = load('{data_file_path}');\n")
        script_file.write(f"result = xcorr(data, {lags}, 'coeff');\n")
        script_file.write(f"save('{result_file_path}', 'result', '-ascii');\n")
        script_file.write("pause(5);\n")

    # Run MATLAB script
    subprocess.run(["/Applications/MATLAB_R2022a.app/bin/matlab", "-batch", f"run('{script_file_path}')"])

    # Load result
    result = np.loadtxt(result_file_path)

    # Clean up temporary files
    shutil.rmtree(session_directory)

    return result








