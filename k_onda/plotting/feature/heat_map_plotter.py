from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from .feature_plotter import FeaturePlotter

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial'] 


class HeatMapPlotter(FeaturePlotter):

    def process_calc(self, calc_config):
        aesthetics = calc_config['aesthetics']
        legend_info_list = calc_config['legend_info_list']

        for figure, colorbar_spec, entries in legend_info_list:
            share = colorbar_spec.get('share')
            location = colorbar_spec.get('location', 'right')

            if share in [None, 'each']:
                for entry in entries:
                    self.process_entries(aesthetics, location, figure, [entry]) 
        
            elif share == 'global':
                self.process_entries(aesthetics, location, figure, entries)
        
            else:
                raise ValueError('Unknown value for share')
        
    def process_entries(self, aesthetics, location, figure, entries):
        norm = self.get_norm(entries)

        for entry in entries:
            axes = []
            for entry in entries:
                img, ax = self.plot_entry(entry, aesthetics, norm)
                axes.append(ax)

            figure.colorbar(img, ax=axes, cax=figure.cax, location=location)
            

    
    def get_marker_args(self, aesthetic_args):
        
        marker_args = {'cmap': 'jet', 'aspect': 'auto', 'origin': 'lower'}
        marker_args.update(aesthetic_args.get('marker', {}))
        return marker_args
  
    def plot_entry(self, entry, aesthetics, norm):
        aesthetic_args = self.get_aesthetic_args(entry, aesthetics)
        marker_args = self.get_marker_args(aesthetic_args)
        ax = entry['cell']
        val = entry[entry['attr']]
        img = ax.imshow(val, norm=norm, **marker_args)
        return img, ax
        
    def get_norm(self, entries):
        attr = entries[0]['attr']
        vmax, vmin = [fun([entry[attr] for entry in entries]) for fun in (np.max, np.min)]
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        return norm
    
      