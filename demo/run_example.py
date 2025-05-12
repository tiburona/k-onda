import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from k_onda.main import run_pipeline



CONFIG_PATH = Path(__file__).parent / "config.json"

import json
import random

# --- Automatically populate demo spike times and update firing rate ---
DURATION_SECONDS = 100  # total recording duration for the demo
FIRING_RATE_HZ = 5      # mean firing rate (spikes per second)
NUM_SPIKES = DURATION_SECONDS * FIRING_RATE_HZ

# Use a fixed seed so the demo is reproducible every time it’s run
random.seed(42)
spike_times = sorted(random.uniform(0, DURATION_SECONDS) for _ in range(NUM_SPIKES))

# Load the JSON config, update the first (and only) demo unit, then write it back
with open(CONFIG_PATH, "r+", encoding="utf-8") as f:
    config = json.load(f)
    unit = config["animals"][0]["units"]["good"][0]
    unit["spike_times"] = [round(t, 4) for t in spike_times]
    unit["firing_rate"] = round(len(spike_times) / DURATION_SECONDS, 2)
    f.seek(0)
    json.dump(config, f, indent=2)
    f.truncate()



psth_plot = {
    'plot_type': 'psth',
    'section': {
        'attr': 'calc',
        'aesthetics': {
            'ax': {'border': {'top': {'visible': 'FFF'}, 'right': {'visible': 'FFF'}}},
            'default': 
                    {'marker': {'color': 'black'},
                    }},
        'label': {'x_bottom': 
                  {'text': 'Seconds'}, 
                  'y_left': {'text': 'Firing Rate (Spikes per Second)'}},        
        'divisions': [
            {
                'divider_type': 'period_type',
                'members': ['stim']
            }]
    }}



PSTH_OPTS = {
    
    'procedure': 'make_plots',
    'plot_spec': psth_plot,
    'write_opts': './psth',
    'calc_opts': {'kind_of_data': 'spike', 'calc_type': 'firing_rates', 
                  'base': 'event', 'bin_size': .01, 
                  'periods': {'stim': {'period_pre_post': (1, 0), 'event_pre_post': (.05, 1)}} 
    }}


run_pipeline(config_file=CONFIG_PATH, opts=PSTH_OPTS)