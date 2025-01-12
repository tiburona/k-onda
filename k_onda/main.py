import sys
import os
import pstats
import signal
import cProfile
import tracemalloc

# Add the parent directory of k_onda to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("sys.path:", sys.path)
print("Current working directory:", os.getcwd())

from k_onda.utils import log_directory_contents
from k_onda.run import Runner

from k_onda.run.misc_data_init.opts_library import  RUNNER_OPTS,  GROUP_PSTH_OPTS, MRL_OPTS, MRL_PREP_OPTS, PSTH_CSV_OPTS, AS_OPTS
from k_onda.run.misc_data_init.CH27_plot_specs import *



def main():
    run()


def run(log=True):
    config_file =  '/Users/katie/likhtik/AS/init_config.json'
    # config_file = ('/Users/katie/likhtik/likhtik_scripts/spike_data_processing/' 
    #               'documentation/tutorials/psth/data/init_config.json')
    # config_file = '/Users/katie/likhtik/CH27mice/init_config.json'
    runner = Runner(config_file=config_file)

    runner.run(AS_OPTS)
 
    if log:
        log_directory_contents('/Users/katie/likhtik/data/logdir')


def timeout_handler(signum, frame):
    raise TimeoutError


def profile_run(timeout=1000):
    # Set the signal handler for the alarm signal
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    profiler = cProfile.Profile()
    try:
        profiler.enable()
        run()
    except TimeoutError:
        print("Profiling stopped due to timeout")
    finally:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumulative')
        stats.print_stats(100)
        profile_filename = os.path.join('/Users/katie/likhtik/data/logdir', 'profile_output.prof')
        with open(profile_filename, 'w') as f:
            stats = pstats.Stats(profiler, stream=f).sort_stats('cumulative')
            stats.print_stats()
        print(f"Profiling results saved to {profile_filename}")


def visualize_profile():
    stats = pstats.Stats('/Users/katie/likhtik/data/logdir/profile_output.prof')

    # Sort the statistics by cumulative time and print the top 10 functions 
    stats.sort_stats('cumulative').print_stats(10)


def memory_profile_run():

    tracemalloc.start()

    try:
        run()

    except MemoryError as e:
       print("MemoryError encountered:", e)
        # Handle the memory error if necessary
    finally:
        # Always take the memory snapshot, even in case of errors
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')

        print("Top 10 memory consuming lines:")
        for stat in top_stats[:10]:
            print(stat)
    

if __name__ == '__main__':
    main()

