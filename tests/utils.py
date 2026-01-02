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


def write_two_region_lfp(
    *,
    duration=40,
    fs=sampling_rate,
    carrier_freq=theta_freq,
    envelope_freq=0.5,
    delay=0.05,
    noise=noise_sd,
    seed=0,
):
    """
    Create a two-channel LFP .nix file with a shared envelope and a fixed lag.

    Channel 0: band-limited carrier with a slow sinusoidal envelope.
    Channel 1: same envelope and carrier shifted by `delay` seconds.
    """
    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs

    rng = np.random.default_rng(seed)

    envelope = 1 + 0.5 * np.sin(2 * np.pi * envelope_freq * t)
    carrier = np.sin(2 * np.pi * carrier_freq * t)
    signal_0 = envelope * carrier

    shifted_t = t - delay
    shifted_envelope = 1 + 0.5 * np.sin(2 * np.pi * envelope_freq * shifted_t)
    signal_1 = shifted_envelope * np.sin(2 * np.pi * carrier_freq * shifted_t)

    signal = np.stack([signal_0, signal_1], axis=1)
    if noise:
        signal += rng.normal(scale=noise, size=signal.shape)

    analogsignal = neo.AnalogSignal(
        signal,
        units="uV",
        sampling_rate=fs * pq.Hz,
        t_start=0 * pq.s,
    )
    segment = neo.Segment(name="simulated_lfp_pair")
    segment.analogsignals.append(analogsignal)

    block = neo.Block()
    block.segments.append(segment)

    tmpdir = tempfile.TemporaryDirectory()
    output_file = Path(tmpdir.name) / "lfp_two_region.nix"

    with neo.io.NixIO(filename=str(output_file), mode="ow") as io:
        io.write_block(block)

    return tmpdir, output_file
