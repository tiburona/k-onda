
from k_onda.utils import log_directory_contents
from k_onda.run import Runner


DEFAULT_CONFIG = '/path/to/init_config.json'


def run_pipeline(config_file=DEFAULT_CONFIG, opts=None, prep=None, logdir=None):
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