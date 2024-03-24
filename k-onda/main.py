from utils import log_directory_contents
from runner import Runner
from misc_data_init.opts_library import TEST_RUNNER_OPTS
import cProfile
import pstats
import signal
import time


def timeout_handler(signum, frame):
    raise TimeoutError


def main():
    runner = Runner(config_file='/Users/katie/likhtik/IG_INED_Safety_Recall/init_config.json', lfp=True)
    # runner.run('make_spreadsheet', TEST_RUNNER_OPTS, path='/Users/katie/likhtik/IG_INED_Safety_Recall',
    #             filename='power_pre_pip_and_behavior')
    runner.run('mrl_bar_plot', TEST_RUNNER_OPTS)
    log_directory_contents('/Users/katie/likhtik/data/logdir')


# Set the signal handler for the alarm signal\
signal.signal(signal.SIGALRM, timeout_handler)
# Schedule the alarm to go off after 5 seconds
signal.alarm(500)

profiler = cProfile.Profile()
try:
    profiler.enable()
    
    while True:
        if __name__ == '__main__':
            main()
    
    profiler.disable()
except TimeoutError:
    profiler.disable()
    print("Profiling stopped due to timeout")

# Print the profiling results
stats = pstats.Stats(profiler).sort_stats('cumulative')
stats.print_stats()







