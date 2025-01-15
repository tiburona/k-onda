import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np

from .feature_plotter import FeaturePlotter
from ..plotting_helpers import format_label
from k_onda.base import Base
from k_onda.utils import safe_get

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial'] 


class LinePlotter(FeaturePlotter):
    
    def plot_row(self, ax, val, aesthetic_args=None):
        ax.plot(np.arange(len(val)), val, **aesthetic_args.get('marker', {}))


class WaveformPlotter(LinePlotter):

    def plot_row(self, ax, val, aethetic_args):
        super().plot_row(ax, val, aethetic_args)
        # do something to label plot?


class HeatMapPlotter(FeaturePlotter):
    
    def plot_row(self, ax, val, row, aesthetic_args=None):
        ax.imshow(val, cmap='jet', aspect='auto')
          

class TextPlotter(Base):
    """
    Very minimal text plotter.
    """
    def plot(self, cell, text_spec):
        # If cell is an AxWrapper, cell.ax is the actual Matplotlib Ax
        ax = cell.ax if hasattr(cell, 'ax') else cell
        ax.text(0.5, 0.5, text_spec['content'], ha='center', va='center')
        ax.set_axis_off()
        


