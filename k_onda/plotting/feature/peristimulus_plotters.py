import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np

from .feature_plotter import FeaturePlotter
from ..plotting_helpers import format_label
from k_onda.utils import safe_get

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial'] 


class HistogramPlotter:
    
    def plot_hist(self, x, y, width, ax, aesthetic_args):
        ax.bar(x, y, width=width, **aesthetic_args.get('marker', {})) 


class PeriStimulusPlotter(FeaturePlotter):
    
    def __init__(self ):
        self.base = self.calc_opts.get('base', 'event') 
        self.pre, self.post = (getattr(self, f"{opt}_{self.base}") for opt in ('pre', 'post'))
        self.marker_names = []
        
    def plot_entry(self, ax, val, aesthetic_args):
            data_divisions, ax_list, x_slices = self.handle_broken_axes(val)
            row = 'foo' # TODO fix this
            for i, (ax, data, _) in enumerate(zip(ax_list, data_divisions, x_slices)):
                self.plot_line(ax, data, row, i, aesthetic_args, data_source=row['data_source'])
                self.place_indicator(ax, aesthetic_args)
                  

    def plot_line(self, ax, val, aesthetic_args):
        data_divisions, ax_list, x_slices = self.handle_broken_axes(val)
        row = 'foo' # TODO fix this
        for i, (ax, data, _) in enumerate(zip(ax_list, data_divisions, x_slices)):
            self.plot_row(ax, data, row, i, aesthetic_args, data_source=row['data_source'])
            self.place_indicator(ax, aesthetic_args)
                
    def set_x_ticks(self, ax, data, x_slice):

        length = len(data[0]) if data.ndim > 1 else len(data)

        nearest_power_of_ten = 10 ** np.floor(np.log10(self.pre + self.post))
        rounded_step = nearest_power_of_ten 
        if 0 < self.pre < rounded_step:
            beginning = 0
        else:
            beginning = -self.pre
        
        manual_ticks = np.arange(beginning, self.post, step=rounded_step) 
        ax.set_xticks(manual_ticks)

        # Get the existing tick positions (in bins)
        existing_ticks = ax.get_xticks()

        # Filter the ticks to only those within the visible range of the data (i.e., corresponding to x_slice)
        visible_ticks = [tick for tick in existing_ticks if 0 <= tick <= length]

        # Calculate the start and end time for the current x_slice (in seconds)
        x_slice_start_time = x_slice[0] - self.pre  
        x_slice_end_time = x_slice[1] - self.pre

        # Create a time range that matches the visible ticks, from x_slice_start_time to x_slice_end_time
        tick_range = np.linspace(x_slice_start_time, x_slice_end_time, len(visible_ticks))

        # Set the x-tick positions and labels
        ax.set_xticks(visible_ticks)  # Use only the visible ticks
        ax.set_xticklabels([f"{label:.2f}" for label in tick_range])  # Labels in seconds
            

    def place_indicator(self, ax, aesthetic_args):
        indicator = aesthetic_args.get('indicator', {})
        if not indicator:
            return
        indicator_type = indicator.get('type')
        when = indicator['when']

        if indicator_type == 'patch':
            if len(when) == 2:
                width = when[1] - when[0]
                # Create a transformation: x in data coords, y in axes coords
                transform = transforms.blended_transform_factory(ax.transData, ax.transAxes)
                ax.add_patch(plt.Rectangle(
                    (when[0], 0), width, 1,
                    facecolor='gray', alpha=0.3,
                    transform=transform  # Apply the transformation
                ))
                

class RasterPlotter(PeriStimulusPlotter):
    
    def process_calc(self, *args, **kwargs):
        super().process_calc(*args, **kwargs)
                
    def plot_line(self, ax, data, row, slice_no, aesthetic_args, data_source=None):
        ax.set_ylim(0, len(data))  # Set ylim based on the number of rows
        line_length = aesthetic_args.get('line_length', .9)   
        marker_args = aesthetic_args.get('marker', {})  
        row_label_spec = safe_get(aesthetic_args, ['label', 'ax', 'row_labels'])
        if data_source and row_label_spec and slice_no == 0:
            if isinstance(row_label_spec, str) and row_label_spec.startswith('lambda'):
                row_label_func = eval(row_label_spec)
                row_labels = row_label_func(row['data_source'])
            else:
                row_labels = format_label(row_label_spec, data_source)
        else:
            row_labels = ['' for _ in range(len(data))]

        # Plot spikes on the vlines
        for i, spiketrain in enumerate(data):
            ax.text(-3, i + line_length / 2, row_labels[i], ha='right', va='center', rotation=45, fontsize=8)  # Adjust x-coordinate to position label
            for j, spike in enumerate(spiketrain):
                if spike:
                    ax.vlines(j, i, i + line_length, **marker_args)
    
                
class PeriStimulusHistogramPlotter(PeriStimulusPlotter, HistogramPlotter):
    
    def plot_entry(self, ax, val, row, aesthetic_args):
        if val.ndim > 1:
            val = val[0]
        x = np.linspace(-self.pre, self.post, len(val)+1)[:-1]
        y = val
        self.plot_hist(x, y, self.calc_opts['bin_size'], ax, aesthetic_args)