import matplotlib.pyplot as plt
import numpy as np
import xarray

from .feature_plotter import FeaturePlotter
from k_onda.utils import safe_get

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial'] 


class LinePlotter(FeaturePlotter):

    def coord_vals(self, val, aesthetic_args):
        """Returns the x-coordinates for the plot. Assumes val is a 1D array."""
        if isinstance(val, xarray.DataArray) or isinstance(val, xarray.Dataset):
            x_coord = safe_get(aesthetic_args, ['tick_labels', 'x', 'x_coord'])
            if x_coord is not None:
                return val.coords[x_coord]
            if hasattr(val, 'coords') and val.coords:
                return val.coords.values()[0].pint.data.magnitude
            else:
                if isinstance(val, xarray.DataArray):
                    return np.arange(len(val))
                else:
                    da = next(iter(val.data_vars.values()))
                    return np.arange(len(da))
        else:
            return np.arange(len(val.data_vars.values()))


    def plot_entry(self, ax, val, aesthetic_args=None):
        ax.plot(self.coord_vals(val, aesthetic_args), val, label='', **aesthetic_args.get('marker', {}))

    # def get_handle(self, entry, index):
    #     return entry['cell'].get_lines()[index]
    
class VerticalLinePlotter(LinePlotter):
    
    def plot_entry(self, ax, val, aesthetic_args=None):
        # Get the keys of the dataset; assume the first key holds the centers 
        # and the second key holds the lengths.
        keys = list(val.data_vars)
        if len(keys) < 2:
            raise ValueError("The dataset must contain at least two data variables: centers and lengths.")
        
        centers = val[keys[0]].values
        lengths = val[keys[1]].values

        if aesthetic_args.get('line_length_multiplier'):
            line_length_multiplier = aesthetic_args['line_length_multiplier']
        elif keys[1] in ['sem', 'std', 'sem_envelope']:
            # we want SEM vertical lines to extend the length of the sem above and below the point
            line_length_multiplier = 1.0
        else:
            # we assume that the lengths provided should be the total length of the line
            line_length_multiplier = 0.5

        # Draw each vertical line with a height equal to the length provided,
        # centered vertically at y_center.
        for coord_val, center, length in zip(self.coord_vals(val, aesthetic_args), centers, lengths):
            ax.vlines(coord_val, 
                      center - length*line_length_multiplier, 
                      center + length*line_length_multiplier,
                      label='', 
                      **(aesthetic_args.get('marker', {}) if aesthetic_args else {}))


class WaveformPlotter(LinePlotter):

    def plot_entry(self, ax, val, aesthetic_args):
        super().plot_entry(ax, val, aesthetic_args)
        # do something to label plot?