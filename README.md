# K-Onda  
_Modular Electrophysiology Analysis Pipeline_

[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/)

K-Onda is a modular Python application for analyzing electrophysiology experiments.  It takes as input curated spike data, raw local field potential (LFP) recordings, and/or behavioral data, and outputs publication-quality plots and structured CSVs for downstream analysis.

The architecture is designed to generalize over time to support other experimental domains. 

**License & usage:** All rights reserved – shared *read-only* for evaluation. Contact Katie Surrence for permission before using this code.


## Minimal Example
```
from k_onda.main import Runner

OPTS = {"your": "options_go_here"}  # Replace with analysis-specific options

runner = Runner(config_file="example_experiment_config.json")
runner.run(opts=OPTS)
```

## Features

- Modular pipeline for spike, LFP, and behavioral data
- Hierarchical object model: Experiment → Animal → (Unit, for spike data) -> Period -> Event -> Time Bin
- Experimental Design: Animals can have Conditions and Periods can be of different Period Types
- Publication-quality plotting engine with customizable layout
- Integrations with Matlab, Phy, and R


<br>
<details>
<summary>Configuration structure & commit helper</summary>

 
We recommend organizing your config files outside the K-Onda directory structure (even though the tool currently just uses a direct config_file= argument). For example:

```
your_workspace/
├── k-onda/                        # main code (this repo)
└── analysis-config-for-k-onda/   # your private configs
```
Use `Runner(config_file=...)` to point to the experiment `config_file`.

This layout is supported by a script to commit both repos (we recommend a commit of both configuration and the K-Onda code every time you perform an analysis):

`./devtools/commit_both.sh "Your commit message"`

If you'd prefer to save your config files elsewhere, edit the path in `commit_both.sh`
</details>

## Setup Instructions

Basic setup requires Python 3.9+ and `pip`. 

On MacOS/Linux:

```
python -m venv .venv && source .venv/bin/activate
```

## Requirements

Some functionality (esp. LFP analysis) requires Matlab — see details below.

<details>
<summary>⚠️ Matlab dependency (for LFP analysis)</summary>

Some core functionality—like calculating power and coherence from raw LFP data—**requires Matlab**.

Specifically:
- A working Matlab installation (tested with Matlab 2022a)
- Scripts from Professor Kenneth Harris's lab:
  - `mtcsg.m`
  - `mtchg.m`
- (Optional) `removeLineNoise_SpectrumEstimation.m` for filtering

These scripts are **not included**.  
If you are in the Likhtik lab, contact the author via WhatsApp.  
Others may request access via email.

</details>



