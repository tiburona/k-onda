from utils import log_directory_contents
from runner import Runner
from misc_data_init.opts_library import RUNNER_OPTS, VALIDATION_DATA_OPTS, MS_26_OPTS
import pstats
import signal
import cProfile
import tracemalloc
import os


def main():
    run()


def run(log=True):
    #runner = Runner(config_file='/Users/katie/likhtik/IG_INED_SAFETY_RECALL/init_config.json')
    runner = Runner(config_file='/Users/katie/likhtik/MS_26/init_config.json')
    runner.run(MS_26_OPTS) 
               #prep={'method': 'validate_lfp_events', 'calc_opts': VALIDATION_DATA_OPTS})
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







