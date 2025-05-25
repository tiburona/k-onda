
from k_onda.utils import log_directory_contents
from k_onda.run import Runner
from k_onda.utils import load_config_py


DEFAULT_CONFIG = '/path/to/init_config.json'
CONFIG = '/Users/katie/likhtik/k-onda-analysis/IG_INED_SAFETY/config/init_config.json'
OPTS = load_config_py('/Users/katie/likhtik/k-onda-analysis/IG_INED_SAFETY/config/ig_ined_safety_power_opts.py')
PREP_OPTS = OPTS.PREP_OPTS
PLOT_OPTS = OPTS.POWER_PLOT_OPTS

def run_pipeline(config_file=CONFIG, opts=PLOT_OPTS, prep=PREP_OPTS, logdir=None):
    """Run the core analysis pipeline using user-supplied options."""
    runner = Runner(config_file=config_file)

    runner.run(opts=opts, prep=prep)

    if logdir:
        log_directory_contents(logdir)

# For profiling tools (e.g., time/memory usage), see k_onda/devtools/debug_utils.py
def main():
    """Entry point for CLI or script execution."""
    run_pipeline()


if __name__ == '__main__':
    main()