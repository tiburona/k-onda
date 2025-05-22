import numpy as np
import quantities as pq
import neo
from pathlib import Path
import tempfile
import shutil

from k_onda.main import run_pipeline
from k_onda.devtools import find_project_root
from k_onda.example_configs import LFP_OPTS

PROJECT_ROOT = find_project_root()

# --- Configuration ---
sampling_rate = 2000  # Hz
duration_sec = 130
n_samples = sampling_rate * duration_sec
theta_freq = 8.0  # Hz
theta_boost = 4.0
noise_sd = 0.5

# Stim periods and events
stim_starts = [20, 40, 60, 80, 100]
event_offsets = [i for i in range(10)]  # 10 events per stim period
event_duration = 0.3  # seconds



def test_smoke_lfp():

    # --- Generate signal ---
    time = np.arange(n_samples) / sampling_rate
    signal = np.random.normal(scale=noise_sd, size=n_samples)

    for stim_start in stim_starts:
        for i in event_offsets:
            event_time = stim_start + i
            start = int(event_time * sampling_rate)
            end = int((event_time + event_duration) * sampling_rate)
            if end > n_samples:
                continue
            signal[start:end] += theta_boost * np.sin(2 * np.pi * theta_freq * time[start:end])

    # --- Save to Neo format ---
    signal = signal.reshape(-1, 1)  # shape (samples, channels)
    analogsignal = neo.AnalogSignal(
        signal,
        units="uV",
        sampling_rate=sampling_rate * pq.Hz,
        t_start=0 * pq.s
    )
    segment = neo.Segment(name="simulated_lfp")
    segment.analogsignals.append(analogsignal)

    block = neo.Block()
    block.segments.append(segment)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "lfp_demo.nix"  
        with neo.io.NixIO(filename=str(output_file), mode="ow") as io:
            io.write_block(block)
            config_file = Path(f"{PROJECT_ROOT}/k_onda/example_configs/config.json")
            out_dir = tempfile.mkdtemp()

            LFP_OPTS["io_opts"]["paths"]["lfp"] = output_file
            LFP_OPTS["io_opts"]["paths"]["out"] = f"{out_dir}/lfp"

        run_pipeline(config_file=config_file, opts=LFP_OPTS)

        # Check that expected output files exist
        output_dir = Path(out_dir)
        assert (output_dir / "lfp.png").exists(), "Missing output: lfp.png"
        assert (output_dir / "lfp.txt").exists(), "Missing output: lfp.txt"

        # Cleanup
        shutil.rmtree(out_dir)

test_smoke_lfp()