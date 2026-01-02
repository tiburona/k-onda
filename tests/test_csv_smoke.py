from copy import deepcopy
from pathlib import Path
import os
import tempfile
from collections import defaultdict

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import matplotlib
matplotlib.use("Agg", force=True)

from k_onda.main import run_pipeline
from k_onda.core.base import Base
from k_onda.resources.devtools import find_project_root
from k_onda.resources.example_configs import LFP_OPTS, PSTH_OPTS
from .utils import write_lfp_files


PROJECT_ROOT = find_project_root()
config_file = Path(f"{PROJECT_ROOT}/k_onda/resources/example_configs/config.json")

def _reset_base_state():
    Base._cache = defaultdict(dict)
    Base._shared_filters = {}
    Base._criteria = defaultdict(lambda: defaultdict(tuple))
    Base._selected_period_type = ''
    Base._selected_period_types = []
    Base._selected_period_group = []
    Base._selected_neuron_type = ''
    Base._selected_brain_region = ''
    Base._selected_frequency_band = ''
    Base._selected_region_set = []
    Base._calc_opts = {}
    Base._io_opts = {}
    Base._experiment = None


def test_smoke_csv():
    _reset_base_state()
    psth_calc_opts = deepcopy(PSTH_OPTS["calc_opts"])
    psth_calc_opts.update(
        {
            "row_type": "spike_period",
            "calc_type": "firing_rates",
            "periods": {
                "stim": {"period_pre_post": (0, 1), "event_pre_post": (0, 1)},
                "prestim": {"period_pre_post": (0, 1), "event_pre_post": (0, 1)},
            },
        }
    )

    lfp_calc_opts = deepcopy(LFP_OPTS["calc_opts"])
    lfp_calc_opts.update(
        {
            "row_type": "lfp_period",
            "periods": psth_calc_opts["periods"],
        }
    )

    csv_opts = {
        "procedure": "make_csv",
        "calc_opts": [psth_calc_opts, lfp_calc_opts],
        "io_opts": {
            "paths": {"out": "foo"},
            "read_opts": {"lfp_file_load": "neo"},
        },
    }

    tmpdir, lfp_input_file = write_lfp_files()
    output_dir = Path(tmpdir.name)
    output_file = Path(tmpdir.name) / "psth_lfp.csv"
    csv_opts["io_opts"]["paths"]["lfp"] = lfp_input_file
    csv_opts["io_opts"]["paths"]["lfp_output"] = str(output_dir)
    csv_opts["io_opts"]["paths"]["out"] = str(output_file)

    run_pipeline(config_file=config_file, opts=csv_opts)

    assert (output_dir / "psth_lfp.csv").exists(), "Missing output: psth_lfp.csv"

    tmpdir.cleanup()
