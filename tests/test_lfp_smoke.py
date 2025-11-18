from pathlib import Path

from k_onda.main import run_pipeline
from k_onda.resources.devtools import find_project_root
from k_onda.resources.example_configs import LFP_OPTS

from utils import write_lfp_files

PROJECT_ROOT = find_project_root()
config_file = Path(f"{PROJECT_ROOT}/k_onda/resources/example_configs/config.json")


def test_smoke_lfp():

   
    tmpdir, input_file = write_lfp_files()
    output_dir = Path(tmpdir.name)
    output_file = Path(tmpdir.name) / "lfp"

    LFP_OPTS["io_opts"]["paths"]["lfp"] = input_file
    LFP_OPTS["io_opts"]["paths"]["lfp_output"] = str(output_dir)
    LFP_OPTS["io_opts"]["paths"]["out"] = str(output_file)

    run_pipeline(config_file=config_file, opts=LFP_OPTS)

    assert (output_dir / "lfp.png").exists(), "Missing output: lfp.png"
    assert (output_dir / "lfp.txt").exists(), "Missing output: lfp.txt"

    tmpdir.cleanup()
 