import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.transforms as transforms
import numpy as np

from .feature_plotter import FeaturePlotter
from .heat_map_plotter import HeatMapPlotter
from k_onda.utils import safe_get

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial'] 





class HistogramPlotter:
    
    def plot_hist(self, x, y, width, ax, aesthetic_args):
        ax.bar(self.to_float(x), self.to_float(y), width=width, **aesthetic_args.get('marker', {})) 


class PeriStimulusPlotter(FeaturePlotter):
    
    def __init__(self ):  
        self.marker_names = []                  
                
    def set_x_ticks(self, ax, data):

        # TODO: we don't have period type set when this is called anymore!  we need
        # another way of figuring out pre and post

        length = len(data[0]) if data.ndim > 1 else len(data)

        nearest_power_of_ten = 10 ** np.floor(np.log10(self.pre + self.post))
        rounded_step = nearest_power_of_ten 
        if 0 < self.pre < rounded_step:
            beginning = 0
        else:
            beginning = -self.pre


        # Get the existing tick positions (in bins)
        existing_ticks = ax.get_xticks()

        # Filter the ticks to only those within the visible range of the data 
        visible_ticks = [tick for tick in existing_ticks if 0 <= tick <= length]

        # Create a time range that matches the visible ticks
        tick_range = np.linspace(beginning, self.post, len(visible_ticks))
        
        manual_ticks = np.arange(beginning, self.post, step=rounded_step) 
        ax.set_xticks(manual_ticks)

       
        # Create a time range that matches the visible ticks
        tick_range = np.linspace(beginning, self.post, len(visible_ticks))
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
                    facecolor='gray', alpha=0.2,
                    transform=transform,  # Apply the transformation
                    zorder=10
                ))

    def split_data_for_break_axes(self, cell, val, break_axes):
        base = self.calc_opts.get('base', 'event')
        x_coords = self.calc_opts.get('x_coords', f'{base}_time')
        x = val.coords[x_coords]

        if not break_axes:
            return ([cell.obj, x, val],)

        if 0 in break_axes:
            brk = break_axes[0]
    
            dim = brk.get('dim', x_coords)

            get_splits = lambda arr: [
                arr.where((arr[dim] >= lower) & (arr[dim] < upper), drop=True) 
                for lower, upper in brk['splits']
            ]

            return zip(cell.ax_list, get_splits(x), get_splits(val))

        return ([cell.obj, x, val],)
                

class RasterPlotter(PeriStimulusPlotter):
        
    def plot_entry(self, cell, val, aesthetic_args, break_axes=None):
       
        for ax, x_split, val_split in self.split_data_for_break_axes(cell, val, break_axes):

            def custom_format(x, pos):
                # x is the tick location (a float)
                idx = int(round(x))
                coord_vals = x_split.coords[f"{self.calc_opts.get('base')}_time"].values
                if 0 <= idx < len(coord_vals):
                    val = coord_vals[idx]
                    if isinstance(val, (float, int)):
                        return f"{val:.1f}"
                    return str(val)
                return ''
            
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(custom_format))

            self.plot_line(ax, val_split, aesthetic_args)
            self.place_indicator(ax, aesthetic_args)
                
    def plot_line(self, ax, data, aesthetic_args):
        line_length = aesthetic_args.get('line_length', .9)   
        marker_args = aesthetic_args.get('marker', {})  
        for j, spike in enumerate(data):
            if spike:
                ax.vlines(j, 0, line_length, **marker_args)
    
                
class PeriStimulusHistogramPlotter(PeriStimulusPlotter, HistogramPlotter):
    
    def plot_entry(self, cell, val, aesthetic_args):
        if val.ndim > 1:
            val = val[0]
        break_axes = getattr(cell, 'break_axes', None)
        for ax, x_split, val_split in self.split_data_for_break_axes(cell, val, break_axes):
            self.plot_hist(x_split, val_split, self.calc_opts['bin_size'], ax, aesthetic_args)
            self.place_indicator(ax, aesthetic_args)


class PeriStimulusHeatMapPlotter(HeatMapPlotter, PeriStimulusPlotter):

    def plot_entry(self, entry, aesthetic_args, norm, break_axes=None):
        if break_axes:
            raise NotImplementedError("Break axes not yet implemented for heat maps")
        img, ax = HeatMapPlotter.plot_entry(self, entry, aesthetic_args, norm)
        self.place_indicator(ax, aesthetic_args)
        return img, ax
    

class PeriStimulusPowerSpectrumPlotter(PeriStimulusHeatMapPlotter):

    def process_calc(self, calc_config):
        super().process_calc(calc_config)

    def get_marker_args(self, aesthetic_args):
        marker_args = super().get_marker_args(aesthetic_args)
        # TODO I need another way of knowing what pre and post event are because 
        # they aren't the current period
        extent = (-self.pre_event, self.post_event, self.freq_range[0], self.freq_range[1])
        marker_args['extent'] = extent
        return marker_args

