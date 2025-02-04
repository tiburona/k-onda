from copy import deepcopy

from matplotlib.gridspec import  GridSpecFromSubplotSpec
import numpy as np

from k_onda.base import Base
from .legend import ColorbarMixin
from .plotting_helpers import reshape_subfigures


class Layout(Base, ColorbarMixin):

    def __init__(self, parent, index, figure=None, processor=None, dimensions=None,
                 gs_args=None):
        self.parent = parent
        self.index = index
        self.figure = figure
        
        self.gs_args = gs_args if gs_args else {}

        self.processor = processor

        if self.processor is None:
            self.processor_type = None
            self.spec = None
        else:
            self.processor_type = processor.name
            self.spec = self.processor.spec

        self.dimensions = dimensions or self.calculate_my_dimensions()

        self.create_grid()

        if self.no_more_processors:
            self.cells = self.make_all_cells()
        else:
            self.cells = self.subfigure_grid
        

    def subfigures(self, nrows, ncols, **kwargs):
        subfigures = self.figure.subfigures(nrows, ncols, **kwargs)

        # If it's a single SubFigure, wrap it in a 2D array
        if not isinstance(subfigures, (list, np.ndarray)):
            return np.full((nrows, ncols), [SubfigWrapper(subfigures, (0, 0), self)])

        elif isinstance(subfigures, np.ndarray) and subfigures.ndim == 1: 
            return np.array([SubfigWrapper(subfig, (nrows, ncols), self) 
                             for subfig in subfigures]).reshape(nrows, ncols)
            
        elif isinstance(subfigures, np.ndarray) and subfigures.ndim == 2:
            return np.array([[SubfigWrapper(subfig, (i, j), self) 
                              for j, subfig in enumerate(row)] 
                              for i, row in enumerate(subfigures)])
            
        else:
            raise ValueError("Unexpected dimensions for subfigures")
            
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
        self.adjust_grid_for_labels()

        if self.global_colorbar:
            self.create_outer_and_subgrid()

        subfigure_args = self.spec.get('subfigure', {}) if self.processor else {}
            
        self.subfigure_grid = self.subfigures(*self.dimensions, **self.gs_args, 
                                              **subfigure_args)
        
        if self.spec and self.spec.get('subplots_adjust'):
            self.figure.get_figure().subplots_adjust(**self.spec['subplots_adjust'])
        

    def adjust_grid_for_labels(self):
        if self.processor and self.processor.label:

            self.label_figure = self.figure

            self.adjust_dimension('title', label_figure_dims=(3, 1), 
                new_figure_ind=(1, 0), ratio_dim='height', rvrs=False)
            
            self.adjust_dimension('x', label_figure_dims=(3, 1), 
                new_figure_ind=(1, 0), ratio_dim='height', rvrs=True)

            self.adjust_dimension('y', label_figure_dims=(1, 3), 
                new_figure_ind=(0, 1), ratio_dim='width', rvrs=False)
         

    def adjust_dimension(self, position, label_figure_dims, new_figure_ind, ratio_dim, rvrs):
        
        position_info = self.processor.label.get(position)

        if not position_info:
            return
        
        space_between = position_info.get('space_between', .05)
        space_within = position_info.get('space_within', .05)
        ratios = [space_between, 1, space_within]
        if rvrs:
            reversed(ratios)

        ratio_dict = {f"{ratio_dim}_ratios": ratios}

        grid = self.subfigures(*label_figure_dims, **ratio_dict)
    
        self.figure = grid[new_figure_ind]
        
    def make_all_cells(self):
        
        return np.array([
            [AxWrapper(self.subfigure_grid[i,j].subfigures(1, 1).add_subplot(), (i, j), self) 
             for j in range(self.dimensions[1])] 
             for i in range(self.dimensions[0])
        ])
    
    @property
    def no_more_processors(self):
        return self.processor and (
            self.processor.name == 'segment' or 
            self.processor.next is None
            )
        
    def make_subfigure(self, i, j):

        if self.colorbar_for_each_plot:
            subfigure = self.colorbar_enabled_subfigure(
                self.figure, self.subfigure_grid[i, j], self.colorbar_position)
        else:
            subfigure = self.figure.add_subfigure(self.subfigure_grid[i, j])
            
        return subfigure
       

class Wrapper(Base):

    def __init__(self, obj, index, layout):
        self.obj = obj
        self.figure = layout.figure # the subfigure the obj is on
        self.index = index 
        self.layout = layout
        self.processor = layout.processor

    def __getattr__(self, name):
        # Forward any unknown attribute access to the original obj
        return getattr(self.obj, name)
    
    @staticmethod
    def is_extreme_index(index, layout, dim, last):
        return index[dim] ==  (layout.dimensions[dim] - 1 if last else 0)
    
    def parent_is_extreme(self, dim, last):
        processors = []
        current_proc = self.processor
        while current_proc is not None:
            if current_proc.parent_processor:
                processors.append([current_proc.index, current_proc.parent_layout])
            current_proc = current_proc.parent_processor
        if any([not self.is_extreme_index(index, layout, dim, last) 
                for index, layout in processors]):
            return False
        return True
    
    def is_extreme_within_parent(self, dim, last):
        if self.processor.spec_type == 'section':
            return True
        else:
            return self.is_extreme_index(self.index, self.layout, dim, last)
    
    def is_in_extreme_position(self, axis, last, absolute):
        dim = int(not axis == 'x')

        is_extreme_within_parent = self.is_extreme_within_parent(dim, last)
        
        if not absolute:
            return is_extreme_within_parent

        return is_extreme_within_parent and self.parent_is_extreme(dim, last)


class SubfigWrapper(Wrapper):

    def __init__(self, subfig, index, layout):
        super().__init__(subfig, index, layout)

class AxWrapper(Wrapper):
    def __init__(self, ax, index, layout):
        super().__init__(ax, index, layout)
    

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


        
      
                        
                
                
    



