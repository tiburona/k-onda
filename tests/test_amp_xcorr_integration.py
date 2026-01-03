import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.signal import correlate, hilbert
from scipy.signal.windows import tukey

from k_onda.core.base import Base
from k_onda.resources.devtools import find_project_root
from k_onda.run.initialize_experiment import Initializer
from k_onda.run.runner import CalcOptsProcessor

from tests.utils import write_two_region_lfp

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


def _build_config(lfp_path):
    config = json.loads(CONFIG_PATH.read_text())
    config["paths"]["lfp"] = str(lfp_path)

    animal = config["animals"][0]
    animal["lfp_electrodes"] = {"bla": 0, "hpc": 1}
    animal["period_info"]["stim"]["onsets"] = [600000]
    animal["period_info"]["stim"]["duration"] = 10
    animal["period_info"]["prestim"]["duration"] = 10

    return config


def _build_calc_opts(calc_type: str):
    calc_opts = {
        "kind_of_data": "lfp",
        "calc_type": calc_type,
        "region_sets": ["bla_hpc"],
        "frequency_bands": [(6.0, 10.0)],
        "lfp_padding": (0, 0),
        "max_lag_sec": 0.25,
        "filters": {"amp_xcorr": {"method": "none"}},
        "periods": {
            "stim": {"event_pre_post": (0.2, 0.2)},
            "prestim": {"event_pre_post": (0.2, 0.2)},
        },
    }
    return CalcOptsProcessor(calc_opts).process()[0]


def _prepare_experiment(calc_opts, lfp_path, tmpdir):
    _reset_base_state()
    initializer = Initializer(_build_config(lfp_path))
    experiment = initializer.init_experiment()
    experiment.calc_opts = calc_opts
    experiment.selected_brain_region = calc_opts["region_set"].split("_")[0]
    experiment.selected_region_set = calc_opts["region_set"]
    experiment.selected_frequency_band = calc_opts["frequency_band"]
    experiment.selected_period_type = "stim"

    output_dir = Path(tmpdir.name)
    experiment.io_opts = {
        "paths": {
            "lfp": lfp_path,
            "lfp_output": str(output_dir),
            "out": str(output_dir / "lfp"),
        },
        "read_opts": {"lfp_file_load": "neo"},
    }
    experiment.lfp_prep()
    return experiment


def _normalized_xcorr_manual(x, y, fs):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.shape != y.shape:
        raise ValueError("x and y must be the same shape")

    n = x.size
    lags = np.arange(-n + 1, n, dtype=int)

    zx = (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) > 0 else 1.0)
    zy = (y - y.mean()) / (y.std(ddof=0) if y.std(ddof=0) > 0 else 1.0)

    raw = correlate(zx, zy, mode="full", method="auto")
    overlap = n - np.abs(lags)
    corr = raw / overlap

    return corr, lags / fs


def _manual_amp_xcorr(calculator):
    fs = calculator.to_float(calculator.lfp_sampling_rate, unit="Hz")
    pad_len = calculator.to_int(calculator.lfp_padding)
    alpha = 0.2

    envelopes = []
    for signal in calculator.padded_regions_data:
        analytic = hilbert(np.asarray(signal))
        start = int(pad_len[0])
        stop = -int(pad_len[1]) if pad_len[1] else None
        env = np.abs(analytic[start:stop])
        env = env * tukey(env.size, alpha=float(alpha))
        envelopes.append(env)

    amp1, amp2 = envelopes

    corr, lags = _normalized_xcorr_manual(amp1, amp2, fs)

    max_lag_sec = calculator.calc_opts.get("max_lag_sec")
    if max_lag_sec is None and "lags" in calculator.calc_opts:
        max_lag_sec = calculator.calc_opts["lags"] / fs

    if max_lag_sec is not None:
        mask = (lags >= -max_lag_sec) & (lags <= max_lag_sec)
        corr = corr[mask]
        lags = lags[mask]

    corr = np.clip(corr, -1 + 1e-12, 1 - 1e-12)
    return xr.DataArray(corr, dims=("lag",), coords={"lag": lags})


def _stim_calculator(experiment):
    return next(
        calc for calc in experiment.all_amp_xcorr_calculators if calc.period_type == "stim"
    )


def test_amp_xcorr_matches_manual_correlation():
    calc_opts = _build_calc_opts(calc_type="amp_xcorr")
    tmpdir, lfp_path = write_two_region_lfp(duration=40, delay=0.05, noise=0.01)

    experiment = _prepare_experiment(calc_opts, lfp_path, tmpdir)
    try:
        calculator = _stim_calculator(experiment)
        manual = _manual_amp_xcorr(calculator)
        aggregated = experiment.get_amp_xcorr()

        np.testing.assert_allclose(np.asarray(aggregated), manual.data)
        np.testing.assert_allclose(np.asarray(aggregated.coords["lag"]), manual.coords["lag"])
    finally:
        tmpdir.cleanup()


def test_lag_of_max_corr_matches_manual_argmax():
    calc_opts = _build_calc_opts(calc_type="lag_of_max_corr")
    tmpdir, lfp_path = write_two_region_lfp(duration=40, delay=0.05, noise=0.01)

    experiment = _prepare_experiment(calc_opts, lfp_path, tmpdir)
    try:
        calculator = _stim_calculator(experiment)
        manual = _manual_amp_xcorr(calculator)
        manual_lag = manual.idxmax("lag").item()

        computed = calculator.get_lag_of_max_corr()
        np.testing.assert_allclose(computed, manual_lag)
    finally:
        tmpdir.cleanup()
