from copy import deepcopy
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from k_onda.main import run_pipeline
from k_onda.resources.devtools import find_project_root
from k_onda.run.runner import CalcOptsProcessor


PROJECT_ROOT = find_project_root()
CONFIG_PATH = Path(f"{PROJECT_ROOT}/k_onda/resources/example_configs/config.json")


def _build_calc_opts():
    calc_opts = {
        "kind_of_data": "spike",
        "calc_type": "spike_counts",
        "base": "event",
        "row_type": "spike_event",
        "time_type": "continuous",
        "bin_size": 0.01,
        "periods": {
            "stim": {"period_pre_post": (0.05, 1), "event_pre_post": (0.05, 1)},
            "prestim": {"period_pre_post": (0.05, 1), "event_pre_post": (0.05, 1)},
        },
    }
    return CalcOptsProcessor(calc_opts).process()[0]


def _load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _all_events_from_config(cfg, sampling_rate):
    animal = cfg["animals"][0]
    period_info = animal["period_info"]

    stim_onsets_raw = period_info["stim"]["onsets"]
    stim_onsets_s = [on / sampling_rate for on in stim_onsets_raw]
    stim_events = [[onset + i for i in range(10)] for onset in stim_onsets_s]

    prestim_shift = period_info["prestim"]["shift"]
    prestim_events = [
        [onset + prestim_shift + i for i in range(10)] for onset in stim_onsets_s
    ]

    periods = []
    for idx, onset in enumerate(stim_onsets_s):
        periods.append(("stim", idx, onset, stim_events[idx]))
        periods.append(("prestim", idx, onset + prestim_shift, prestim_events[idx]))

    return periods, animal


def _manual_spike_counts_df(calc_opts):
    cfg = _load_config()
    sampling_rate = cfg["sampling_rate"]
    spikes = np.array(cfg["animals"][0]["units"]["good"][0]["spike_times"])

    periods, animal_cfg = _all_events_from_config(cfg, sampling_rate)

    pre, post = calc_opts["periods"]["stim"]["event_pre_post"]
    bin_size = calc_opts["bin_size"]
    num_bins = int(round((pre + post) / bin_size))

    rows = []
    for period_type, period_id, period_onset, events in periods:
        period_start = period_onset
        for event_id, ev_time in enumerate(events):
            start = ev_time - pre
            stop = ev_time + post
            counts, _ = np.histogram(spikes, bins=num_bins, range=(start, stop))
            for time_bin, count in enumerate(counts):
                absolute_time = np.round(start + time_bin * bin_size, 8)
                relative_time = np.round(absolute_time - ev_time, 8)
                period_time = np.round(absolute_time - period_start, 8)
                rows.append(
                    {
                        "spike_counts": float(count),
                        "experiment": cfg["identifier"],
                        "group": cfg["group_names"][0],
                        "unit": animal_cfg["identifier"] + "_good_1",
                        "period": period_id,
                        "event": event_id,
                        "time_bin": time_bin,
                        "absolute_time": absolute_time,
                        "relative_time": relative_time,
                        "period_time": period_time,
                        "event_time": relative_time,
                        "period_type": period_type,
                        "category": "good",
                        "neuron_type": animal_cfg["units"]["good"][0]["neuron_type"],
                        "quality": animal_cfg["units"]["good"][0]["quality"],
                        "animal": animal_cfg["identifier"],
                    }
                )

    return pd.DataFrame(rows)


def test_spike_counts_csv_matches_manual_histogram():
    calc_opts = _build_calc_opts()
    tmpdir = tempfile.TemporaryDirectory()
    output_file = Path(tmpdir.name) / "spike_counts.csv"

    opts = {
        "procedure": "make_csv",
        "calc_opts": [deepcopy(calc_opts)],
        "io_opts": {"paths": {"out": str(output_file)}},
    }

    run_pipeline(config_file=CONFIG_PATH, opts=opts)

    csv_df = pd.read_csv(output_file, comment="#")

    manual_df = _manual_spike_counts_df(calc_opts)
    manual_df["spike_counts"] = manual_df["spike_counts"].astype(int)
    manual_df = manual_df[csv_df.columns]

    sort_cols = ["experiment", "unit", "period", "event", "time_bin"]
    csv_df_sorted = csv_df.sort_values(sort_cols).reset_index(drop=True)
    manual_df_sorted = manual_df.sort_values(sort_cols).reset_index(drop=True)

    pd.testing.assert_frame_equal(csv_df_sorted, manual_df_sorted)

    tmpdir.cleanup()
