from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from .feature_plotter import FeaturePlotter

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial'] 


class HeatMapPlotter(FeaturePlotter):

    def process_calc(self, calc_config):
        info = calc_config['info']
        aesthetics = calc_config['aesthetics']

        colorbar_spec = aesthetics.pop('color_bar_spec', {})
        share = colorbar_spec.get('share', None)
        location = colorbar_spec.get('location', 'right')

        if share in [None, 'each']:
            for entry in info:
                self.process_entries([entry], aesthetics, location) 
        
        elif share == 'global':
            self.process_entries(info, aesthetics, location)

        elif share in [0, 1, 'row', 'col']:
            # if you select 0, rows share a color bar, i.e., entries in info 
            # whose 0th digit in index is the same share a color bar.
            # if you select 1, columns share a color bar, i.e., entries in info
            # whose 1st digit in index is the same share a color bar.

            info_matrix = self.reshape_info_to_2d(info)

            if share in [1, 'col']: 
                info_matrix = info_matrix.T

            for row in info_matrix:
                self.process_entries(row, aesthetics, location)
        
        else:
            raise ValueError('Unknown value for share')
        
    def process_entries(self, entries, aesthetics, location):
        norm = self.get_norm(entries)
        axes = []
        imgs = [] 
        
        for entry in entries:
            

            img, ax = self.plot_entry(entry, aesthetics, norm)
            axes.append(ax)
            imgs.append(img)

        fig = axes[0].figure
        fig.colorbar(imgs[-1], ax=axes, location=location)

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
    
    @staticmethod
    def reshape_info_to_2d(info):
        """
        Reshape a list of dictionaries 'info' into a 2D array (NumPy),
        where 'info[i]['index'] = (row, col)' indicates the position.
        
        Each element of the returned 2D array is the corresponding dictionary
        from 'info'.
        """
        # 1. Find the max row and column indices
        max_i = max(d['index'][0] for d in info) + 1
        max_j = max(d['index'][1] for d in info) + 1
        
        # 2. Create an empty NumPy array of shape (max_i, max_j), storing objects
        arr = np.empty((max_i, max_j), dtype=object)
        
        # 3. Place each dictionary in its 2D position
        for d in info:
            i, j = d['index']
            arr[i, j] = d

        return arr
      