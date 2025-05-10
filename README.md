# k-onda  
_modular electrophysiology analysis pipeline_

[![Python](https://img.shields.io/badge/python-3.9+-blue)](https://www.python.org/)

**License & usage:** All rights reserved – shared *read-only* for evaluation.
Contact Katie Surrence for permission before using this code in any project.

K-Onda is an applicatsion for analyzing electrophysiology experiments in Python, and in time it will be extended/abstracted to other kinds of data.  It takes as input curated spike data, raw local field potential data, and/or behavioral data and outputs plots and/or csv files for further data analysis. 


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

<br>
<details>
<summary>Configuration structure & commit helper</summary>

 
We recommend saving your config files outside the K-Onda directory structure. For example:

```
your_workspace/
├── k-onda/                        # main code (this repo)
└── analysis-config-for-k-onda/   # your private configs
```
Use `Runner(config_file=...)` to point to the experiment `config_file`.

This layout is supported by a script to commit both repos (we recommend a commit of both configuration and the K-Onda code every time you perform an analysis):

./devtools/commit_both.sh "Your commit message"

If you'd prefer to save your config files elsewhere, edit the path in `commit_both.sh`
</details>


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



