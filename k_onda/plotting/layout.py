from copy import deepcopy

from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np

from k_onda.base import Base
    

class Layout(Base):

    def __init__(self, parent, index, figure=None, processor=None, dimensions=None,
                 gs_args=None):
        self.parent = parent
        self.index = index
        self.figure = figure
        self.gs_args = gs_args

        self.processor = processor

        if self.processor is None:
            self.processor_type = None
            self.spec = None
        else:
            self.processor_type = processor.name
            self.spec = self.processor.spec

        self.dimensions = dimensions or self.calculate_my_dimensions()
        self.gs = self.create_grid()  # Create the gridspec for the actual data
        self.cells = self.make_all_cells()
            
    @property
    def one_d_cell_list(self):
        return [c for r in self.cells for c in r]
    
    def calculate_my_dimensions(self):
        dims = [1, 1]
        if self.spec is not None:
            if self.processor_type == 'container':
                dims = self.spec['dimensions']

            else: 
                for division in self.spec['divisions']:
                    if 'dim' in division:
                        dims[division['dim']] = len(division['members'])
        return deepcopy(dims)
    
    def create_grid(self):
        gs_args = self.gs_args or dict(left=0.1, right=0.9, top=0.9, bottom=0.1)
        data_gridspec = GridSpec(*self.dimensions, **gs_args)
        return data_gridspec
        
    def make_all_cells(self):
        return np.array([
            [self.make_cell(i, j) for j in range(self.dimensions[1])] 
            for i in range(self.dimensions[0])
        ])
        
    def make_cell(self, i, j):
        if self.processor is None:
            subfigure = self.figure.add_subfigure(self.gs[0, 0])
            return subfigure

        elif self.processor.next is not None and any(
            name in self.processor.next for name in ['section', 'split', 'components']):
            subfigure = self.figure.add_subfigure(self.gs[i, j])
            return subfigure

        else:
            gridspec_slice = self.gs[i, j]
            ax = self.figure.add_subplot(gridspec_slice, zorder=0)
            return AxWrapper(ax, (i, j))

    def add_ax(self, sub_fig_cell, index):
            ax = sub_fig_cell.add_subplot()
            return AxWrapper(ax, index)
    
    

class AxWrapper(Base):

    def __init__(self, ax, index):
        self.ax = ax  # Store the original ax
        self.ax_list = [self]
        self.index = index
        self.bottom_edge = None
        self.left_edge = None
        
    def __getattr__(self, name):
        # Forward any unknown attribute access to the original ax
        return getattr(self.ax, name)
    

class BrokenAxes(Base):
    
    def __init__(self, fig, parent_gridspec, index, break_axes, aspect=None):
        self.break_axes = {
            key: [np.array(t) for t in value] 
            for key, value in break_axes.items()}
        self.index = index
        self.fig = fig
        self.aspect = aspect

        dim0_breaks = len(self.break_axes.get(1, [])) or 1
        dim1_breaks = len(self.break_axes.get(0, [])) or 1

        self.gs = GridSpecFromSubplotSpec(
            dim0_breaks, dim1_breaks, 
            subplot_spec=parent_gridspec[self.index],  
        )

        self.axes, self.ax_list = self._create_subplots()

        self._share_axes_and_hide_spines()

    def _create_subplots(self):
        axes = []
        ax_list = []
        for i0 in range(len(self.break_axes.get(1, [0]))):
            row = []
            for j1 in range(len(self.break_axes.get(0, [0]))):
                gridspec_slice = self.gs[i0, j1]
                ax = self.fig.add_subplot(gridspec_slice)
                if self.aspect:
                    ax.set_box_aspect(self.aspect)
                ax_wrapper = AxWrapper(ax, (i0, j1))
                row.append(ax_wrapper)
                ax_list.append(ax_wrapper)
            axes.append(row)
        return axes, ax_list

    def _share_axes_and_hide_spines(self):
        # Share axes
        for i, dim_num in zip((0, 1), ('y', 'x')):
            if i in self.break_axes:
                first, *rest = self.ax_list
                for ax in rest:
                    getattr(ax, f"share{dim_num}")(first.ax)

        # Hide spines and add diagonal lines to indicate breaks
        d = .015  # size of diagonal lines
        kwargs = dict(color='k', clip_on=False)

        for (dim, dim_num, (first_side, last_side)) in zip(
            ('y', 'x'), (0, 1), (('right', 'left'), ('bottom', 'top'))):
            if dim_num in self.break_axes:
                first, *rest, last = self.ax_list

                # Set spine visibility
                first.spines[first_side].set_visible(False)
                first.tick_params(**{'axis': dim, 'which':'both', first_side: False, 
                                     f"label{first_side}": False})

                last.spines[last_side].set_visible(False)
                last.tick_params(**{'axis': dim, 'which':'both', last_side: False,
                                 f"label{last_side}": False})

                for ax in rest:
                    ax.spines[first_side].set_visible(False)
                    ax.spines[last_side].set_visible(False)
                    ax.tick_params(**{'axis': dim, 'which':'both', first_side: False, last_side: False,
                                      f"label{first_side}": False, f"label{last_side}": False})

                # Add diagonal break markers
                self._add_break_marker(first, first_side, d, **kwargs)
                self._add_break_marker(last, last_side, d, **kwargs)
                for ax in rest:
                    self._add_break_marker(ax, first_side, d, **kwargs)
                    self._add_break_marker(ax, last_side, d, **kwargs)

    def _add_break_marker(self, ax, side, d, **kwargs):
        coords = {
            'right': [(1-d, 1+d), (-d, +d)],  
            'left': [(-d, +d), (-d, +d)],    
            'bottom': [(-d, +d), (-d, +d)],  
            'top': [(-d, +d), (1-d, 1+d)]    
        }
        x_vals, y_vals = coords[side]
        ax.plot(x_vals, y_vals, transform=ax.transAxes, **kwargs)
        
        # Depending on side, adjust the second set of coordinates (diagonal line placement)
        # if side in ('right', 'left'):
        #     ax.plot(x_vals, (1-d, 1+d), transform=ax.transAxes, **kwargs)  # Second y for vertical sides
        # elif side in ('top', 'bottom'):
        #     ax.plot((-d, +d), (1-d, 1+d), transform=ax.transAxes, **kwargs)  # Second x for horizontal sides


        
      
                        
                
                
    



