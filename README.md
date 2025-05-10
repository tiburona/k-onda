K-Onda is an application for analyzing electrophysiology experiments in Python, and in time it will be extended/abstracted to other kinds of data.  It takes as input curated spike data, raw local field potential data, and/or behavioral data and outputs plots and/or csv files for further data analysis. 


## Setup Instructions

Basic setup requires Python 3.7+ and `pip`. 

For a quick setup:
1. Create a virtual environment: `python -m venv env`
2. Activate the virtual environment:
   - On macOS/Linux: `source env/bin/activate`
   - On Windows: `env\Scripts\activate`
3. Install dependencies: `pip install -r requirements.txt`

If you're new to Python or need more detailed instructions, refer to the [Python Environment Setup Guide](https://realpython.com/python-virtual-environments-a-primer/).

## Execution

```
from k_onda.main import Runner

OPTS = {'example_opt': 'val'}

runner = Runner(config_file="example_experiment_config.json")
runner.run(opts=OPTS)

```


## Requirements

In addition to the Python packages specified in requirements.txt, some functionality depends on having Matlab installed on your computer.  Calculations of power and coherence from raw LFP data depend on having a working version of Matlab installed, along with two scripts from Professor Kenneth Harris's lab (`mtcsg.m` and `mtchg.m`) and their dependencies. You can also choose to filter your LFP data using the Matlab script `removeLineNoise_SpectrumEstimation.m`. These scripts are not distributed with this repo; if you need them, please email the author if you are someone reading this who is not in the Likhtik lab, or if you are, Whatsapp her. This functionality was tested with Matlab 2022a.


<details>
<summary>🧩 Configuration structure & commit helper</summary>

This project expects analysis-specific configuration files to live **outside** the main repo.  
We recommend this layout:

your_workspace/
├── k-onda/                        # main code (this repo)
└── analysis-config-for-k-onda/   # your private configs

Use `Runner(config_root=...)` to point to your config directory.

To commit both repos together:

./devtools/commit_both.sh "Your commit message"
</details>
