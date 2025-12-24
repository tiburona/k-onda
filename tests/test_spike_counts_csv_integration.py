from copy import deepcopy
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from k_onda.main import run_pipeline
from k_onda.resources.devtools import find_project_root
from k_onda.run.initialize_experiment import Initializer
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


def _prepare_experiment(calc_opts):
    initializer = Initializer(CONFIG_PATH)
    experiment = initializer.init_experiment()
    experiment.calc_opts = calc_opts
    experiment.spike_prep()
    return experiment


def _manual_spike_counts_df(experiment):
    rows = []

    for event in experiment.all_spike_events:
        unit = event.unit
        period = event.parent
        animal = unit.animal

        spikes_sec = unit.to_float(unit.spike_times, unit="second")
        start_s = event.to_float(event.start, unit="second")
        stop_s = event.to_float(event.stop, unit="second")
        num_bins = event.num_bins_per

        counts, _ = np.histogram(spikes_sec, bins=num_bins, range=(start_s, stop_s))

        group = getattr(animal, "group", None)
        group_id = group.identifier if group is not None else None

        for i, count in enumerate(counts):
            rows.append(
                {
                    "spike_counts": float(count),
                    "experiment": experiment.identifier,
                    "group": group_id,
                    "unit": unit.identifier,
                    "period": period.identifier,
                    "event": event.identifier,
                    "time_bin": i,
                    "period_type": event.period_type,
                    "category": unit.category,
                    "neuron_type": unit.neuron_type,
                    "quality": unit.quality,
                    "animal": animal.identifier,
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

    experiment = _prepare_experiment(calc_opts)
    manual_df = _manual_spike_counts_df(experiment)

    manual_df = manual_df[csv_df.columns]

    sort_cols = ["experiment", "unit", "period", "event", "time_bin"]
    csv_df_sorted = csv_df.sort_values(sort_cols).reset_index(drop=True)
    manual_df_sorted = manual_df.sort_values(sort_cols).reset_index(drop=True)

    pd.testing.assert_frame_equal(csv_df_sorted, manual_df_sorted)

    tmpdir.cleanup()
