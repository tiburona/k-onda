import cProfile
from utils import log_directory_contents
from procs import *

"""
The top-level module for calling spike processing functions. 
"""


def main():
    plot_cross_correlations_by_unit_pair() # An example.  See what else is available in the procs module.
    log_directory_contents('/Users/katie/likhtik/data/logdir')


if __name__ == '__main__':
    main()
