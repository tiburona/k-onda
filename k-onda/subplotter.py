import numpy as np
from plotter_base import PlotterBase
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


class Figurer(PlotterBase):
            
    def __init__(self):
        self.fig = None
        self.gs = None

    def make_fig(self):
        self.fig = plt.figure(constrained_layout=True)
        self.active_fig = self.fig
        self.gs = GridSpec(1, 1, figure=self.fig)
        self.subplotter = Subplotter(self, [0, 0], first=True)
        self.active_plotter = self.subplotter
        self.active_ax = self.active_plotter.axes[0, 0]
        return self.active_fig
    

class Subplotter(PlotterBase):

    def __init__(self, parent, index, spec=None, first=False, aspect=None, dimensions=None, 
                 grid_keywords=None, invisible_axes=None):
        self.fig = self.active_fig
        self.parent = parent
        self.index = index
        self.spec = spec
        self.first = first
        self.aspect = aspect
        self._dimensions = dimensions
        self.grid_keywords = grid_keywords
        self.gs = self.create_grid()
        self._axes = None
        if first:
            self._ax_visibility = np.array([[False]])
        else:
            self._ax_visibility = np.array(
                [[True if (i, j) not in invisible_axes else False 
                  for j in range(self.dimensions[1])] 
                  for i in range(self.dimensions[0])])
        
    @property
    def axes(self):
        if self._axes is None:
            self._axes = self.make_all_axes()
        return self._axes
    
    @property
    def dimensions(self):
        if self._dimensions is None:
            self._dimensions = self.calculate_my_dimensions()
        return self._dimensions
    
    @property
    def ax_visibility(self):
        return self._ax_visibility
    
    @ax_visibility.setter
    def ax_visibility(self, visibility):
        self._ax_visibility = visibility

    def calculate_my_dimensions(self):
        dims = [1, 1]
        if self.first:
            return dims
        for division in self.spec.values():
            if 'dim' in division:
                dims[division['dim']] = len(division['members'])
        return dims
        
    def create_grid(self):
        grid_keywords = self.grid_keywords if self.grid_keywords else {}
        return GridSpecFromSubplotSpec(
            *self.dimensions, 
            subplot_spec=self.parent.gs[*self.index], 
            **grid_keywords)
        
    def make_all_axes(self):
        return np.array([
            [self.make_ax(i, j) for j in range(self.dimensions[1])] for i in range(self.dimensions[0])
            ])
    
    def make_ax(self, i, j):
        gridspec_slice = self.gs[i, j]
        ax = self.fig.add_subplot(gridspec_slice)
        ax.set_visible(self.ax_visibility[i, j])
        if self.aspect:
            ax.set_box_aspect(self.aspect)
        ax.index = (i, j)
        return ax
    
    def apply_aesthetics(self, aesthetics):
        for key, val in aesthetics.get('border', {}).items():
            spine, tick, label = (val[i] in ['T', True, 'True'] for i in range(3))
            self.active_ax.spines[key].set_visible(spine)
            self.active_ax.tick_params(**{f"label{key}":label, key:tick})

