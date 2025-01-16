import matplotlib.pyplot as plt
import numpy as np

from .feature_plotter import FeaturePlotter

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial'] 


class LinePlotter(FeaturePlotter):
    
    def plot_row(self, ax, val, aesthetic_args=None):
        ax.plot(np.arange(len(val)), val, **aesthetic_args.get('marker', {}))


class WaveformPlotter(LinePlotter):

    def plot_row(self, ax, val, aesthetic_args):
        super().plot_row(ax, val, aesthetic_args)
        # do something to label plot?