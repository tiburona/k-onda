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

The file that you execute to make one of those things happen is [main.py](main.py). The `main` function initializes a `Runner`, which is initialized with the configuration of your experiment.  You then call `run` with the configuration of the procedure you're going to run.  The two configurations can be paths to JSON files or Python dictionaries.  

Here is an example `main` function that plots a peri-stimulus time histogram:
```
def main():
    runner = Runner(config_file='<path_to_config>/init_config.json')
    runner.run(PSTH_OPTS)
```
Here experiment is configured in a JSON file accessible at the given path, and that the analysis configuration is a Python dictionary defined in the same module or imported from another. 


## Requirements

In addition to the Python packages specified in requirements.txt, some functionality depends on having Matlab installed on your computer.  Calculations of power and coherence from raw LFP data depend on having a working version of Matlab installed, along with two scripts from Professor Kenneth Harris's lab (`mtcsg.m` and `mtchg.m`) and their dependencies. You can also choose to filter your LFP data using the Matlab script `removeLineNoise_SpectrumEstimation.m`. These scripts are not distributed with this repo; if you need them, please email the author if you are someone reading this who is not in the Likhtik lab, or if you are, Whatsapp her. This functionality was tested with Matlab 2022a.


## How to use the package

Documentation is very much a work in process, and right now, for demonstration purposes, it is limited to a tutorial on how to produce a PSTH plot from the output of spike sorting.  







