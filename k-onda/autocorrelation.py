from functools import partial
import concurrent.futures
import pandas as pd
import numpy as np
import os
import subprocess
import uuid
import shutil

from utils import cache_method


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


# This is a top-level function which can be pickled
def get_all_autocorrelations(child, opts, method, neuron_type):
    return child.get_all_autocorrelations(opts, method, neuron_type)


class AutocorrelationNode:

    def __init__(self):
        pass

    @cache_method
    def _autocorr_np(self, x, max_lag, demean=False):
        return self.trim_result(np.correlate(x, x, mode='full'), max_lag)

    @cache_method
    def _autocorr_pd(self, x, max_lag, demean=False):
        return np.array([pd.Series(x).autocorr(lag=lag) for lag in range(max_lag + 1)])[1:]

    @cache_method
    def _autocorr_ml(self, x, max_lag):
        return self.trim_result(xcorr(x, max_lag), max_lag)

    @staticmethod
    def trim_result(result, max_lag):
        mid = result.size // 2
        return result[mid + 1:mid + max_lag + 1] / result[mid]

    def _calculate_autocorrelation(self, rates, opts, method, demean=False):
        if not len(rates):
            return np.array([])
        result = getattr(self, f"_autocorr_{method}")(rates, opts['max_lag'])
        if demean:
            result = result - np.mean(result)
        return result

    @staticmethod
    def _avg_autocorrs(autocorrs):
        return np.mean([autocorr for autocorr in autocorrs if not np.all(np.isnan(autocorr))], axis=0)

    def get_all_autocorrelations(self, opts, method, neuron_type=None, demean=False):
        ac_results = {}
        # Calculate the autocorrelation over rates for this node
        rates = self.get_average(opts, 'get_trials_rates', neuron_type=neuron_type)
        demeaned_rates = rates - np.mean(rates)

        ac_results[f"{self.name}_by_rates"] = self._calculate_autocorrelation(demeaned_rates, opts, method,
                                                                              demean=demean)

        # Calculate the autocorrelation over children for this node
        # Remember that the type of autocorrelation computed over children depends on the type of the children,
        # so we need to ask each child to calculate its autocorrelations first.
        children_autocorrs = [child.get_all_autocorrelations(opts, method, neuron_type=neuron_type, demean=demean)
                              for child in self.children]
        for key in children_autocorrs[0]:  # Assuming all children have the same autocorrelation keys
            ac_results[f"{self.name}_by_{key}"] = self._avg_autocorrs(
                [child_autocorrs[key] for child_autocorrs in children_autocorrs])
        return ac_results





