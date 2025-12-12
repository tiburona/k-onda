from copy import deepcopy
from pathlib import Path

from k_onda.main import run_pipeline
from k_onda.resources.devtools import find_project_root
from k_onda.resources.example_configs import LFP_OPTS, PSTH_OPTS
from .utils import write_lfp_files



PSTH_CALC_OPTS = PSTH_OPTS['calc_opts']
PSTH_CALC_OPTS['row_type'] = 'spike_period'
PSTH_CALC_OPTS['calc_type'] = 'firing_rates'
PSTH_CALC_OPTS['periods'] =  {'stim': {'period_pre_post': (0, 1), 'event_pre_post': (0, 1)},
                              'prestim': {'period_pre_post': (0, 1), 'event_pre_post': (0, 1)}}

LFP_CALC_OPTS = LFP_OPTS['calc_opts']
LFP_CALC_OPTS['row_type'] = 'lfp_period'
LFP_CALC_OPTS['periods'] = PSTH_CALC_OPTS['periods']


PROJECT_ROOT = find_project_root()
config_file = Path(f"{PROJECT_ROOT}/k_onda/resources/example_configs/config.json")


CSV_OPTS = {
    'procedure': 'make_csv',
    'calc_opts': [PSTH_CALC_OPTS, LFP_CALC_OPTS],
    'io_opts': {
        'paths': {'out': 'foo'},
        'read_opts': {'lfp_file_load': 'neo'}
        }
}


def test_smoke_csv():
    tmpdir, lfp_input_file = write_lfp_files()
    output_dir = Path(tmpdir.name)
    output_file = Path(tmpdir.name) / "psth_lfp.csv"
    CSV_OPTS["io_opts"]["paths"]["lfp"] = lfp_input_file
    CSV_OPTS["io_opts"]["paths"]["lfp_output"] = str(output_dir)
    CSV_OPTS["io_opts"]["paths"]["out"] = str(output_file)

    run_pipeline(config_file=config_file, opts=CSV_OPTS)

    assert (output_dir / "psth_lfp.csv").exists(), "Missing output: psth_lfp.csv"

    tmpdir.cleanup()
