from utils import log_directory_contents
from runner import Runner
from misc_data_init.opts_library import VALIDATION_DATA_OPTS, GRAPH_OPTS, SPREADSHEET_OPTS, RUNNER_OPTS
import cProfile
import pstats
import signal


def timeout_handler(signum, frame):
    raise TimeoutError


def main():
    run()


def run():
    runner = Runner(config_file='/Users/katie/likhtik/CH_EXT/init_config.json',
                    lfp=True)
    runner.run('make_spreadsheet', SPREADSHEET_OPTS, 
               path='/Users/katie/likhtik/CH_EXT', filename='phase_phase_il_bf')
               # prep={'method': 'validate_events', 'data_opts': VALIDATION_DATA_OPTS})
    #runner.run('plot_power', RUNNER_OPTS)
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







