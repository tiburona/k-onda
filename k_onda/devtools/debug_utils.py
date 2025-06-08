from collections import Counter
import os
import pstats
import signal
import cProfile
import tracemalloc

from k_onda.main import run_pipeline, DEFAULT_LOGDIR


def timeout_handler(signum, frame):
    raise TimeoutError


def profile_run(timeout=1000):
    """Profile the run with a timeout limit."""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    profiler = cProfile.Profile()
    try:
        profiler.enable()
        run_pipeline()
    except TimeoutError:
        print("Profiling stopped due to timeout")
    finally:
        profiler.disable()
        profile_path = os.path.join(DEFAULT_LOGDIR, 'profile_output.prof')
        with open(profile_path, 'w') as f:
            stats = pstats.Stats(profiler, stream=f).sort_stats('cumulative')
            stats.print_stats()
        print(f"Profiling results saved to {profile_path}")


def visualize_profile():
    """Display cumulative time profile statistics."""
    stats = pstats.Stats(os.path.join(DEFAULT_LOGDIR, 'profile_output.prof'))
    stats.sort_stats('cumulative').print_stats(10)


def memory_profile_run():
    """Run the pipeline and report memory usage."""
    tracemalloc.start()
    try:
        run_pipeline()
    except MemoryError as e:
        print("MemoryError encountered:", e)
    finally:
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        print("Top 10 memory consuming lines:")
        for stat in top_stats[:10]:
            print(stat)


