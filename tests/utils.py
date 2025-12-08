import numpy as np
import neo
import tempfile
from pathlib import Path
import quantities as pq


# --- Configuration defaults ---
sampling_rate = 2000  # Hz
duration_sec = 130
theta_freq = 8.0  # Hz
theta_boost = 4.0
noise_sd = 0.5

# Stim periods and events
stim_starts = [20, 40, 60, 80, 100]
event_offsets = [i for i in range(10)]  # 10 events per stim period
event_duration = 0.3  # seconds

def write_lfp_files(
    *,
    duration=duration_sec,
    stim_period_starts=stim_starts,
    event_offsets_per_period=event_offsets,
    fs=sampling_rate,
    theta=theta_freq,
    theta_scale=theta_boost,
    noise=noise_sd,
):
    """
    Create a synthetic LFP .nix file with oscillatory bursts at each event.

    Parameters mirror the module-level defaults so tests can shrink the data
    footprint without touching the smoke-test configuration.
    """
    n_samples = int(fs * duration)
    time = np.arange(n_samples) / fs
    signal = np.random.normal(scale=noise, size=n_samples)

    for stim_start in stim_period_starts:
        for i in event_offsets_per_period:
            event_time = stim_start + i
            start = int(event_time * fs)
            end = int((event_time + event_duration) * fs)
            if end > n_samples:
                continue
            signal[start:end] += theta_scale * np.sin(2 * np.pi * theta * time[start:end])

    # --- Save to Neo format ---
    signal = signal.reshape(-1, 1)  # shape (samples, channels)
    analogsignal = neo.AnalogSignal(
        signal,
        units="uV",
        sampling_rate=fs * pq.Hz,
        t_start=0 * pq.s,
    )
    segment = neo.Segment(name="simulated_lfp")
    segment.analogsignals.append(analogsignal)

    block = neo.Block()
    block.segments.append(segment)

    tmpdir = tempfile.TemporaryDirectory() 
    output_file = Path(tmpdir.name) / "lfp_demo.nix"  
    with neo.io.NixIO(filename=str(output_file), mode="ow") as io:
        io.write_block(block)
    
    return tmpdir, output_file
