
from k_onda.utils import log_directory_contents, load_config_py
from k_onda.run import Runner


DEFAULT_CONFIG = '/path/to/init_config.json'

SAFETY_CONFIG = '/Users/katie/likhtik/k-onda-analysis/IG_INED_Safety/config/init_config.json'
POWER_OPTS = '/Users/katie/likhtik/k-onda-analysis/IG_INED_Safety/config/ig_ined_safety_power_opts.py'
PY_OPTS = load_config_py(POWER_OPTS)
PREP = PY_OPTS.PREP_OPTS
#OPTS = PY_OPTS.POWER_PLOT_OPTS
OPTS = PY_OPTS.GROUP_CHECK_OPTS



def run_pipeline(config_file=SAFETY_CONFIG, opts=OPTS, prep=PREP, logdir=None):
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
