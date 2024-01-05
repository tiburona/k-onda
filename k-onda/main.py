from utils import log_directory_contents
from runner import Runner
from misc_data_init.opts_library import TEST_RUNNER_OPTS

"""
The top-level module for calling spike processing functions. 
"""


def main():
    runner = Runner(config_file='/Users/katie/likhtik/CH_for_katie_less_conservative/init_config.json', lfp=True)
    runner.run('mrl_bar_plot', TEST_RUNNER_OPTS)
    log_directory_contents('/Users/katie/likhtik/data/logdir')


if __name__ == '__main__':
    main()
