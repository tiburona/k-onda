from utils import log_directory_contents
from runner import Runner
from misc_data_init.opts_library import TEST_RUNNER_OPTS, VALIDATION_DATA_OPTS
import cProfile
import pstats
import signal


def timeout_handler(signum, frame):
    raise TimeoutError


def main():
    run()


def run():
    runner = Runner(config_file='/Users/katie/likhtik/IG_INED_Safety_Recall/init_config.json') 
                    #lfp=True, behavior=True)
    runner.run('make_spreadsheet', TEST_RUNNER_OPTS, 
               path='/Users/katie/likhtik/IG_INED_Safety_Recall',
               filename='psth')
               #prep=('validate_events', VALIDATION_DATA_OPTS))
    #runner.run('plot_coherence_over_frequencies', TEST_RUNNER_OPTS, prep=('validate_events', VALIDATION_DATA_OPTS))
    log_directory_contents('/Users/katie/likhtik/data/logdir')


def profile_run(timeout=1000):
    # Set the signal handler for the alarm signal\
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
        stats.print_stats()


if __name__ == '__main__':
    main()







