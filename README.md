# K-Onda  
_Modular Electrophysiology Analysis Pipeline_

[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/)
[![CI](https://github.com/tiburona/k-onda/actions/workflows/ci.yml/badge.svg)](https://github.com/tiburona/k-onda/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/tiburona/k-onda/branch/main/graph/badge.svg)](https://codecov.io/gh/tiburona/k-onda)

K-Onda is a Python application for analyzing electrophysiology experiments.  It takes as input curated spike recordings, local field potential (LFP) recordings, and/or behavioral data, and outputs publication-quality plots and structured CSVs for downstream analysis. It performs calculations such as firing rate, auto- and cross-correlations of firing rates, LFP power, amplitude cross correlation, coherence, granger causality between brain regions, and mean resultant length (MRL) calculations of the relationship between LFP and firing rate.

The architecture is designed to generalize over time to support other kinds of data. 

**License & usage:** All rights reserved – shared read-only for evaluation purposes. Contact Katie Surrence for permission before using this code.

## Minimal Example
```
from k_onda.main import Runner

OPTS = {"your": "options_go_here"}  # Replace with analysis-specific options

runner = Runner(config_file="example_experiment_config.json")
runner.run(opts=OPTS)
```

## Demo

A small demo showing how to generate a peristimulus time histogram of firing rates is included in the demo directory. You can run it via the Python script (demo.py), the Jupyter notebook, or [directly on Binder](https://mybinder.org/v2/gh/tiburona/k-onda/HEAD?filepath=demo/k_onda_demo.ipynb).

A Binder tip:
<details>
If the notebook loading bar seems like it's hanging for a long time on yellow, check the raw logs, and scroll down to the end.  If it says "Done", you can hit 
refresh and the notebook will load.
</details>


## Background & Goals

K-Onda developed organically to support flexible, reproducible analysis of electrophysiology experiments in a behavioral neuroscience setting. Many existing tools support a small suite
of analyses, but a goal for K-Onda is to evolve to be extensible to arbitrary analyses and data
types, and there are no existing electrophysiology tools designed to support complex multipanel
layouts, such that you can move from raw data to output for publication with a single application.

Some publications to which K-Onda has contributed figures include:

Fernandes-Henriques C.M., Guetta Y.\*, Sclar M.G.\*, Zhang R., Surrence K., Miura Y.,
Friedman A.K., Likhtik E. (2025) Infralimbic projections to the basal forebrain constrain defensive behavior during extinction learning. Journal of Neuroscience (In press).

Grunfeld I.S.\*, Surrence K.R.\* , Denholtz L.E., Nahmoud I., Hanif S., Burghardt N.S., Likhtik E.
Chronic stress impairs safety cue discrimination, decreases inhibitory firing in the prelimbic
cortex, and tunes prelimbic activity to amygdala theta oscillations. In Preparation.

## Features

- Modular pipeline for spike, LFP, and behavioral data
- Hierarchical object model: Experiment → Animal → (Unit, for spike data) -> Period -> Event -> Time Bin
- Experimental Design: Animals can have Conditions and Periods can be of different Period Types
- Publication-quality plotting engine with customizable layout of multipanel plots.
- Integrations with Matlab and Phy.


<br>
<details>
<summary>Configuration structure & commit helper</summary>

 
We recommend organizing your config files and data output outside the K-Onda directory structure. For example:

```
your_workspace/
├── k-onda/                        # main code (this repo)
└── k-onda-analysis/               # your private configs and data
```
Use `Runner(config_file=...)` to point to the experiment `config_file`.

This layout is supported by a script to commit two or more repos.  

`./scripts/commit_all.sh "Your commit message"`

If you'd prefer to save your config files and data files elsewhere, or save them in separate directories, edit the path in `commit_all.sh`

Experiment configuration is ideally pretty static, but analysis configuration changes frequently.  If you want to both be able to write your analysis configuration in Python, rather than dealing with JSON, and store your analysis configuration in a separate repository, K-Onda allows you to import from outside the repository like so:

```
# analysis_config.py
SOME_SETTING = "alpha"
```

```
config = load_config_py("../my-configs/analysis_config.py")
print(config.SOME_SETTING)
```

</details>

## Setup Instructions

Basic setup requires Python 3.11+ and `pip`. 

On MacOS/Linux:

```
python -m venv .venv && source .venv/bin/activate
```

## Requirements

Some functionality (esp. LFP analysis) requires Matlab — see details below.

<details>
<summary>⚠️ Matlab dependency (for LFP analysis)</summary>

Some functionality gives you the option to use Matlab routines, and for some 
functionality (for instance, Granger causality), it is still required.

In order to use the Matlab functionality, you need
- A working Matlab installation (tested with Matlab 2022a)

and the scripts the particular function calls.  

- Some that are not publically available include scripts from Professor Kenneth Harris's lab:
  - `mtcsg.m`
  - `mtchg.m`
  - `removeLineNoise_SpectrumEstimation.m` for filtering

These scripts are **not included**.  
If you are in the Likhtik lab, contact the author via WhatsApp.  
Others may request access via email.

</details>

## Status

K-Onda is in active development.

