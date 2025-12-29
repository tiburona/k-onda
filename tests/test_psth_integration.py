from copy import deepcopy
import importlib
from pathlib import Path

import numpy as np

import k_onda.resources.example_configs.psth_opts as psth_opts
from k_onda.resources.devtools import find_project_root
from k_onda.run.initialize_experiment import Initializer
from k_onda.run.runner import CalcOptsProcessor

PROJECT_ROOT = find_project_root()
CONFIG_PATH = Path(f"{PROJECT_ROOT}/k_onda/resources/example_configs/config.json")


def _build_calc_opts(base: str):
    """Return a concrete calc_opts entry for the requested base level."""
    calc_opts = deepcopy(importlib.reload(psth_opts).PSTH_OPTS["calc_opts"])
    calc_opts["base"] = base
    return CalcOptsProcessor(calc_opts).process()[0]


def _prepare_experiment(calc_opts):
    """Instantiate an Experiment configured for spike PSTH calculations."""
    initializer = Initializer(CONFIG_PATH)
    experiment = initializer.init_experiment()
    experiment.calc_opts = calc_opts
    experiment.selected_period_type = "stim"
    experiment.spike_prep()
    return experiment


def _event_bounds(event):
    start = event.to_float(event.start, unit="second")
    stop = event.to_float(event.stop, unit="second")
    return start, stop


def _period_bounds(period):
    start = period.to_float(period.start, unit="second")
    stop = period.to_float(period.stop, unit="second")
    return start, stop


def _firing_rates_for_window(start_s, stop_s, spike_times_sec, bin_size):
    num_bins = int(np.rint((stop_s - start_s) / bin_size))
    in_window = spike_times_sec[
        (spike_times_sec >= start_s) & (spike_times_sec <= stop_s)
    ]
    counts, _ = np.histogram(in_window, bins=num_bins, range=(start_s, stop_s))
    return counts / bin_size


def _event_firing_rates(event, spike_times_sec, bin_size):
    start, stop = _event_bounds(event)
    return _firing_rates_for_window(start, stop, spike_times_sec, bin_size)


def _period_firing_rates(period, spike_times_sec, bin_size):
    start, stop = _period_bounds(period)
    return _firing_rates_for_window(start, stop, spike_times_sec, bin_size)


def _unit_firing_std(periods, spike_times_sec, bin_size, base):
    if base == "event":
        windows = [
            _event_firing_rates(event, spike_times_sec, bin_size)
            for period in periods
            for event in period.events
        ]
    elif base == "period":
        windows = [
            _period_firing_rates(period, spike_times_sec, bin_size)
            for period in periods
        ]
    else:
        raise ValueError(f"Unknown base {base!r}")

    if not windows:
        return np.nan

    concatenated = np.concatenate(windows)
    return np.std(concatenated)


def _manual_event_psth(experiment):
    unit = experiment.all_units[0]
    periods = unit.select_children("spike_periods")
    spike_times_sec = unit.to_float(unit.spike_times, unit="second")
    bin_size = experiment.to_float(experiment.bin_size, unit="second")
    firing_std = _unit_firing_std(periods, spike_times_sec, bin_size, base="event")

    psth_rows = []
    for period in periods:
        reference_rates = np.mean(
            [
                _event_firing_rates(ev, spike_times_sec, bin_size)
                for ev in period.reference.events
            ],
            axis=0,
        )
        for event in period.events:
            event_rates = _event_firing_rates(event, spike_times_sec, bin_size)
            psth_rows.append((event_rates - reference_rates) / firing_std)

    return np.mean(psth_rows, axis=0)


def _manual_period_psth(experiment):
    unit = experiment.all_units[0]
    periods = unit.select_children("spike_periods")
    spike_times_sec = unit.to_float(unit.spike_times, unit="second")
    bin_size = experiment.to_float(experiment.bin_size, unit="second")
    firing_std = _unit_firing_std(periods, spike_times_sec, bin_size, base="period")

    psth_rows = []
    for period in periods:
        stim_rates = _period_firing_rates(period, spike_times_sec, bin_size)
        reference_rates = _period_firing_rates(
            period.reference, spike_times_sec, bin_size
        )
        psth_rows.append((stim_rates - reference_rates) / firing_std)

    return np.mean(psth_rows, axis=0)


def test_psth_base_event_matches_manual():
    calc_opts = _build_calc_opts(base="event")
    experiment = _prepare_experiment(calc_opts)

    aggregated = experiment.get_psth()
    manual = _manual_event_psth(experiment)

    np.testing.assert_allclose(np.asarray(aggregated), manual, atol=1e-12)


def test_psth_base_period_matches_manual():
    calc_opts = _build_calc_opts(base="period")
    experiment = _prepare_experiment(calc_opts)

    aggregated = experiment.get_psth()
    manual = _manual_period_psth(experiment)

    np.testing.assert_allclose(np.asarray(aggregated), manual, atol=1e-12)
