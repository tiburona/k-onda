import tempfile
import shutil
from k_onda.main import run_pipeline
from k_onda.devtools import find_project_root
from k_onda.example_configs import PSTH_OPTS
from pathlib import Path



PROJECT_ROOT = find_project_root()

def test_run_pipeline_demo_config():
    config_file = Path(f"{PROJECT_ROOT}/k_onda/example_configs/config.json")
    out_dir = tempfile.mkdtemp()

    # Patch the output path
    PSTH_OPTS["io_opts"]["paths"]["out"] = f"{out_dir}/psth"

    run_pipeline(config_file=config_file, opts=PSTH_OPTS)

    # Check that expected output files exist
    output_dir = Path(out_dir)
    assert (output_dir / "psth.png").exists(), "Missing output: psth.png"
    assert (output_dir / "psth.txt").exists(), "Missing output: psth.txt"

    # Cleanup
    shutil.rmtree(out_dir)

test_run_pipeline_demo_config()