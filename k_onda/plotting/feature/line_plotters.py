import matplotlib.pyplot as plt
import numpy as np

from .feature_plotter import FeaturePlotter

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial'] 


class LinePlotter(FeaturePlotter):

    def coord_vals(self, val):
        """Returns the x-coordinates for the plot. Assumes val is a 1D array."""
        if isinstance(val, np.ndarray):
            return np.arange(len(val))
        elif hasattr(val, 'coords'):
            # Try to determine the first dimension name.
            if hasattr(val, 'dims'):
                dims = val.dims
                if isinstance(dims, (tuple, list)):
                    first_dim = dims[0]
                else:
                    # dims is a mapping (e.g. dict-like), so get the first key.
                    first_dim = list(dims.keys())[0]
                if first_dim in val.coords:
                    return val.coords[first_dim].values
            # Fallback: use the first key in the coords dictionary.
            keys = list(val.coords.keys())
            if keys:
                return val.coords[keys[0]].values
            else:
                raise ValueError("No coordinates found in the provided xarray object.")
        else:
            raise ValueError("Unsupported data type for coord_vals.")
    
    def plot_entry(self, ax, val, aesthetic_args=None):
        ax.plot(self.coord_vals(val), val, label='', **aesthetic_args.get('marker', {}))

    def get_handle(self, ax):
        return ax.get_lines()
    
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
        elif keys[1] in ['sem', 'std']:
            # we want SEM vertical lines to extend the length of the sem above and below the point
            line_length_multiplier = 1.0
        else:
            # we assume that the lengths provided should be the total length of the line
            line_length_multiplier = 0.5

        # Draw each vertical line with a height equal to the length provided,
        # centered vertically at y_center.
        for coord_val, center, length in zip(self.coord_vals(val), centers, lengths):
            ax.vlines(coord_val, 
                      center - length*line_length_multiplier, 
                      center + length*line_length_multiplier,
                      label='', 
                      **(aesthetic_args.get('marker', {}) if aesthetic_args else {}))


class WaveformPlotter(LinePlotter):

    def plot_entry(self, ax, val, aesthetic_args):
        super().plot_entry(ax, val, aesthetic_args)
        # do something to label plot?