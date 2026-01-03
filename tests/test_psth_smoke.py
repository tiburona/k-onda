import os
from collections import defaultdict
from copy import deepcopy
import tempfile
import shutil
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import matplotlib
matplotlib.use("Agg", force=True)

from k_onda.main import run_pipeline
from k_onda.core.base import Base
from k_onda.resources.devtools import find_project_root
from k_onda.resources.example_configs import PSTH_OPTS

PROJECT_ROOT = find_project_root()


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

def test_run_pipeline_demo_config():
    _reset_base_state()
    config_file = Path(f"{PROJECT_ROOT}/k_onda/resources/example_configs/config.json")
    out_dir = tempfile.mkdtemp()

    # Patch the output path
    opts = deepcopy(PSTH_OPTS)
    opts["io_opts"]["paths"]["out"] = f"{out_dir}/psth"

    run_pipeline(config_file=config_file, opts=opts)

    # Check that expected output files exist
    output_dir = Path(out_dir)
    assert (output_dir / "psth.png").exists(), "Missing output: psth.png"
    assert (output_dir / "psth.txt").exists(), "Missing output: psth.txt"

    # Cleanup
    shutil.rmtree(out_dir)
