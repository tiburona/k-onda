from utils import log_directory_contents
from runner import Runner
from misc_data_init.opts_library import VALIDATION_DATA_OPTS, SPREADSHEET_OPTS, \
    BLA_HPC_COHERENCE_RUNNER_OPTS, HPC_PL_COHERENCE_RUNNER_OPTS, BLA_PL_COHERENCE_RUNNER_OPTS
import pstats
import signal


def timeout_handler(signum, frame):
    raise TimeoutError


def main():
    run()


def run():
    runner = Runner(config_file='/Users/katie/likhtik/IG_INED_SAFETY_RECALL/init_config.json',
                    lfp=True)
    runner.run('make_spreadsheet', SPREADSHEET_OPTS, 
                path='/Users/katie/likhtik/IG_INED_SAFETY_RECALL', filename='granger_behavior',
                prep={'method': 'validate_events', 'data_opts': VALIDATION_DATA_OPTS})    
    runner.run('plot_coherence', BLA_PL_COHERENCE_RUNNER_OPTS,
                prep={'method': 'validate_events', 'data_opts': VALIDATION_DATA_OPTS})
    runner.run('plot_coherence', BLA_HPC_COHERENCE_RUNNER_OPTS,
                prep={'method': 'validate_events', 'data_opts': VALIDATION_DATA_OPTS}),
    runner.run('plot_coherence', HPC_PL_COHERENCE_RUNNER_OPTS,
                prep={'method': 'validate_events', 'data_opts': VALIDATION_DATA_OPTS}),
    
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
        stats.print_stats()


if __name__ == '__main__':
    main()







