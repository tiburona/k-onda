from copy import deepcopy
from pathlib import Path
from collections import defaultdict

import numpy as np
import mne

from k_onda.core.base import Base
from k_onda.resources.devtools import find_project_root
from k_onda.resources.example_configs.lfp_opts import SPECTRUM_OPTS
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


def _build_calc_opts(base: str):
    """
    Generate a concrete calc_opts entry (no loop lists) for the requested base.
    We shrink the frequency grid to keep the test fast and aggregate over
    both time and frequency to make comparisons straightforward.
    """
    calc_opts = deepcopy(SPECTRUM_OPTS)
    calc_opts.update(
        {
            "base": base,
            "frequency_type": "block",
            "time_type": "block",
        }
    )

    # Narrow the multitaper grid to speed up the power computation
    calc_opts["power_arg_set"] = deepcopy(calc_opts["power_arg_set"])
    freqs = np.arange(6, 12, 2)
    calc_opts["power_arg_set"].update(
        {
            "freqs": freqs,
            "n_cycles": freqs * 0.5,
        }
    )

    return CalcOptsProcessor(calc_opts).process()[0]


def _prepare_experiment(calc_opts):
    """
    Spin up an Experiment with fresh synthetic data and the supplied calc_opts.
    Returns the experiment and the TemporaryDirectory so callers can clean up.
    """
    np.random.seed(0)
    _reset_base_state()
    tmpdir, input_file = write_lfp_files()
    output_dir = Path(tmpdir.name)

    initializer = Initializer(CONFIG_PATH)
    experiment = initializer.init_experiment()
    experiment.selected_brain_region = calc_opts["brain_regions"][0]
    experiment.selected_frequency_band = calc_opts["frequency_band"]
    experiment.io_opts = {
        "paths": {
            "lfp": input_file,
            "lfp_output": str(output_dir),
            "out": str(output_dir / "lfp"),
        },
        "read_opts": {"lfp_file_load": "neo"},
    }
    experiment.calc_opts = calc_opts
    experiment.lfp_prep()
    return experiment, tmpdir


def _manual_block_power(experiment, base: str):
    """Compute block-averaged power using raw mne multitaper, outside k_onda aggregation."""

    def _as_seconds(q):
        """Extract plain seconds from a pint-xarray quantity."""
        if hasattr(q, "pint"):
            return float(np.asarray(q.pint.to("second").values))
        return float(q)

    calc_opts = experiment.calc_opts
    power_args = calc_opts["power_arg_set"]
    freq_band = calc_opts["frequency_band"]
    tol = calc_opts.get("frequency_tolerance", 0.2)

    period_means = []
    event_means = []

    for period in experiment.all_lfp_periods:
        data = period.padded_data
        data = data if not isinstance(data, dict) else data[period.selected_brain_region]
        data_3d = data[np.newaxis, np.newaxis, :]

        power = mne.time_frequency.tfr_array_multitaper(data_3d, **power_args).squeeze()
        freqs = power_args["freqs"]
        dt = power_args["decim"] / power_args["sfreq"]
        times = np.arange(power.shape[-1]) * dt

        fmask = (freqs >= freq_band[0] - tol) & (freqs <= freq_band[1] + tol)
        power = power[fmask]

        def window_mean(start_s, stop_s):
            mask = (times >= start_s) & (times < stop_s)
            if not mask.any():
                return np.nan
            return power[:, mask].mean(axis=1).mean()

        if base == "period":
            period_means.append(window_mean(0, _as_seconds(period.duration)))
            continue

        pre = _as_seconds(period.pre_event)
        post = _as_seconds(period.post_event)
        for ev in period.event_starts_in_period_time:
            ev_s = _as_seconds(ev)
            event_means.append(window_mean(ev_s - pre, ev_s + post))

    if base == "period":
        return np.nanmean(period_means)
    return np.nanmean(event_means)


def test_power_base_event_matches_event_mean():
    calc_opts = _build_calc_opts(base="event")
    experiment, tmpdir = _prepare_experiment(calc_opts)
    try:
        aggregated = experiment.get_power()
        manual = _manual_block_power(experiment, base="event")
        np.testing.assert_allclose(np.asarray(aggregated), manual)
    finally:
        tmpdir.cleanup()


def test_power_base_period_matches_period_mean():
    calc_opts = _build_calc_opts(base="period")
    experiment, tmpdir = _prepare_experiment(calc_opts)
    try:
        aggregated = experiment.get_power()
        manual = _manual_block_power(experiment, base="period")
        np.testing.assert_allclose(np.asarray(aggregated), manual)
    finally:
        tmpdir.cleanup()
