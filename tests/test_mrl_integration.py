from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from k_onda.resources.devtools import find_project_root
from k_onda.core.base import Base
from k_onda.run.initialize_experiment import Initializer
from k_onda.run.runner import CalcOptsProcessor

from tests.utils import write_lfp_files


PROJECT_ROOT = find_project_root()
CONFIG_PATH = Path(f"{PROJECT_ROOT}/k_onda/resources/example_configs/config.json")


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


def _build_calc_opts():
    """
    Create a concrete calc_opts dict for MRL calculations.
    We keep the periods unpadded to keep spike/sample alignment simple.
    """
    calc_opts = {
        "kind_of_data": "mrl",
        "calc_type": "mrl",
        "brain_regions": ["bla"],
        "frequency_bands": [(6.0, 10.0)],
        "lfp_padding": (0, 0),
        "periods": {
            "stim": {"period_pre_post": (0, 0)},
            "prestim": {"period_pre_post": (0, 0)},
        },
    }
    return CalcOptsProcessor(calc_opts).process()[0]


def _prepare_experiment(calc_opts):
    """
    Spin up an Experiment wired for MRL with synthetic LFP data.
    """
    np.random.seed(0)
    _reset_base_state()
    tmpdir, input_file = write_lfp_files()
    output_dir = Path(tmpdir.name)

    initializer = Initializer(CONFIG_PATH)
    experiment = initializer.init_experiment()
    experiment.calc_opts = calc_opts
    experiment.selected_brain_region = calc_opts["brain_regions"][0]
    experiment.selected_frequency_band = calc_opts["frequency_band"]
    experiment.io_opts = {
        "paths": {
            "lfp": input_file,
            "lfp_output": str(output_dir),
            "out": str(output_dir / "mrl"),
        },
        "read_opts": {"lfp_file_load": "neo"},
    }
    experiment.mrl_prep()
    return experiment, tmpdir


def _bandpass(data, fs, low, high, order=8):
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, data)


def _manual_mrl(calculator):
    """
    Recompute MRL directly from spikes and filtered LFP without k_onda helpers.
    """
    fs = calculator.to_float(calculator.lfp_sampling_rate, unit="Hz")
    low, high = (float(f) for f in calculator.calc_opts["frequency_band"])

    lfp = calculator.period.padded_data
    if isinstance(lfp, dict):
        lfp = lfp[calculator.selected_brain_region]
    filtered = _bandpass(np.asarray(lfp), fs, low, high)

    pad = np.asarray(calculator.to_int(calculator.lfp_padding))
    start = int(pad[0]) if pad.size else 0
    stop = -int(pad[1]) if pad.size > 1 and pad[1] else None
    phases = np.angle(hilbert(filtered))[start:stop]

    onset_s = calculator.to_float(calculator.period.onset, unit="second")
    duration_s = calculator.to_float(calculator.duration, unit="second")
    pre_post = calculator.calc_opts.get("periods", {}).get(calculator.period_type, {})
    pre, post = pre_post.get("period_pre_post", (0, 0))
    start_s = onset_s - pre
    stop_s = onset_s + duration_s + post

    spikes_s = calculator.to_float(
        calculator.unit.spike_times, unit="second", convert_to_scalar=False
    )
    spikes_s = spikes_s[(spikes_s >= start_s) & (spikes_s <= stop_s)]
    spike_idx = np.rint((spikes_s - onset_s) * fs).astype(int)

    n_samples = int(np.rint(duration_s * fs))
    weights = np.bincount(spike_idx, minlength=n_samples).astype(float)
    weights[weights == 0] = np.nan

    length = min(len(phases), len(weights))
    phases = phases[:length]
    weights = weights[:length]

    valid = np.isfinite(phases) & np.isfinite(weights)
    if not valid.any():
        return np.nan

    weight_sum = np.nansum(weights[valid])
    if weight_sum < 2:
        return np.nan

    sx = np.nansum(weights[valid] * np.cos(phases[valid]))
    sy = np.nansum(weights[valid] * np.sin(phases[valid]))
    return np.sqrt(sx**2 + sy**2) / weight_sum


def test_mrl_matches_manual_calculation():
    calc_opts = _build_calc_opts()
    experiment, tmpdir = _prepare_experiment(calc_opts)
    try:
        manual_vals = []
        for calculator in experiment.all_mrl_calculators:
            manual = _manual_mrl(calculator)
            manual_vals.append(manual)
            computed = np.asarray(calculator.get_mrl())
            np.testing.assert_allclose(computed, manual, rtol=1e-6, atol=1e-6)

        aggregated_manual = np.nanmean(manual_vals)
        aggregated = np.asarray(
            experiment.get_average("get_mrl", stop_at="mrl_calculator")
        )
        np.testing.assert_allclose(aggregated, aggregated_manual, rtol=1e-6, atol=1e-6)
    finally:
        tmpdir.cleanup()
