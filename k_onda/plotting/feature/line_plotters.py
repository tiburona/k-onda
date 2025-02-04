import matplotlib.pyplot as plt
import numpy as np

from .feature_plotter import FeaturePlotter

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial'] 


class LinePlotter(FeaturePlotter):
    
    def plot_entry(self, ax, val, aesthetic_args=None):
        ax.plot(np.arange(len(val)), val, label='', **aesthetic_args.get('marker', {}))

    def get_handles(self, ax):
        return ax.get_lines()

class WaveformPlotter(LinePlotter):

    def plot_entry(self, ax, val, aesthetic_args):
        super().plot_entry(ax, val, aesthetic_args)
        # do something to label plot?