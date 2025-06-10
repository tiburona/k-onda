from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from .feature_plotter import FeaturePlotter

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial'] 


class HeatMapPlotter(FeaturePlotter):

    def process_calc(self, calc_config):
        self.set_values(calc_config)
      
        entry_counter = 0
        for figure, colorbar_spec, entries in self.legend_info_list:
            share = colorbar_spec.get('share')
            location = colorbar_spec.get('location', 'right')

            if share in [None, 'each']:
                for entry in entries:
                    self.process_entries(location, figure, [entry], entry_counter) 
        
            elif share == 'global':
                self.process_entries(location, figure, entries, entry_counter)
        
            else:
                raise ValueError('Unknown value for share')
        
    def process_entries(self, location, figure, entries, entry_counter):
        norm = self.get_norm(entries)

        for entry in entries:
            entry_counter +=1
            aesthetic_args = self.get_aesthetic_args(entry)
            ax_args = aesthetic_args.get('ax', {})
            cbar_args = aesthetic_args.get('colorbar', {})
            axes = []
            for entry in entries:
                img, ax = self.plot_entry(entry, aesthetic_args, norm)
                self.apply_ax_args(ax, ax_args, entry_counter)
                axes.append(ax)

            cbar = figure.colorbar(img, ax=axes, cax=figure.cax, location=location)
            if 'tick_labels' in cbar_args:
                self.apply_tick_label_formatting(cbar.ax, cbar_args['tick_labels'])

    
    def get_marker_args(self, aesthetic_args):
        
        marker_args = {'cmap': 'jet', 'aspect': 'auto', 'origin': 'lower'}
        marker_args.update(aesthetic_args.get('marker', {}))
        return marker_args
  
    def plot_entry(self, entry, aesthetic_args, norm):
        marker_args = self.get_marker_args(aesthetic_args).copy()

        aspect = marker_args.pop('aspect', None)  # pull out if present

        ax  = entry['cell']
        val = entry[entry['attr']]                

        x_dim, y_dim = val.dims[:2]

        img = val.plot.imshow(
            ax=ax,
            x=x_dim,
            y=y_dim,
            add_colorbar=False,
            add_labels=False,
            norm=norm,
            **marker_args
        )

        if aspect is not None:
            ax.set_aspect(aspect)

        return img, ax
        
    def get_norm(self, entries):
        attr = entries[0]['attr']
        vmax, vmin = [fun([entry[attr] for entry in entries]) for fun in (np.nanmax, np.nanmin)]
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        return norm
    
      