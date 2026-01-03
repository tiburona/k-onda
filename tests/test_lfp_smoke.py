import os
from collections import defaultdict
from copy import deepcopy
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import matplotlib
matplotlib.use("Agg", force=True)

from k_onda.main import run_pipeline
from k_onda.core.base import Base
from k_onda.resources.devtools import find_project_root
from k_onda.resources.example_configs import LFP_OPTS
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

def test_smoke_lfp():
    _reset_base_state()
    tmpdir, input_file = write_lfp_files()
    output_dir = Path(tmpdir.name)
    output_file = Path(tmpdir.name) / "lfp"

    lfp_opts = deepcopy(LFP_OPTS)
    lfp_opts["io_opts"]["paths"]["lfp"] = input_file
    lfp_opts["io_opts"]["paths"]["lfp_output"] = str(output_dir)
    lfp_opts["io_opts"]["paths"]["out"] = str(output_file)

    prev_mpl_configdir = os.environ.get("MPLCONFIGDIR")
    os.environ["MPLCONFIGDIR"] = str(output_dir)
    run_pipeline(config_file=config_file, opts=lfp_opts)
    if prev_mpl_configdir is not None:
        os.environ["MPLCONFIGDIR"] = prev_mpl_configdir
    else:
        os.environ.pop("MPLCONFIGDIR", None)

    assert (output_dir / "lfp.png").exists(), "Missing output: lfp.png"
    assert (output_dir / "lfp.txt").exists(), "Missing output: lfp.txt"

    tmpdir.cleanup()
 
