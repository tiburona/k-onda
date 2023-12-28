from utils import log_directory_contents
from runner import Runner
from opts_library import TEST_RUNNER_OPTS

"""
The top-level module for calling spike processing functions. 
"""


def main():
    runner = Runner('plot_psth', TEST_RUNNER_OPTS,
                    config_file='/Users/katie/likhtik/CH_for_katie_less_conservative/init_config.json')
    runner.run()
    log_directory_contents('/Users/katie/likhtik/data/logdir')


if __name__ == '__main__':
    main()
