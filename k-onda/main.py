from utils import log_directory_contents
from runner import Runner
from misc_data_init.opts_library import RUNNER_OPTS
import pstats
import signal
import cProfile
import os


def timeout_handler(signum, frame):
    raise TimeoutError


def main():
    profile_run()

def visualize_profile():
    stats = pstats.Stats('/Users/katie/likhtik/data/logdir/profile_output.prof')

    # Sort the statistics by cumulative time and print the top 10 functions
    stats.sort_stats('cumulative').print_stats(10)


def run():
    runner = Runner(config_file='/Users/katie/likhtik/IG_INED_SAFETY_RECALL/init_config.json')
    runner.run('plot_psth', RUNNER_OPTS)
    
    log_directory_contents('/Users/katie/likhtik/data/logdir')


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


if __name__ == '__main__':
    main()







