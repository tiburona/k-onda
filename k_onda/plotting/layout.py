from copy import deepcopy

from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np

from k_onda.base import Base
from .legend import ColorbarMixin

class Layout(Base, ColorbarMixin):

    def __init__(self, parent, index, figure=None, processor=None, dimensions=None,
                 gs_args=None):
        self.parent = parent
        self.index = index
        self.figure = figure
        print("in Layout init")
        print(figure)
        
        self.gs_args = gs_args

        self.processor = processor
        if self.processor:
            print([division['members'] for division in self.processor.spec['divisions']])
        print("")

        if self.processor is None:
            self.processor_type = None
            self.spec = None
        else:
            self.processor_type = processor.name
            self.spec = self.processor.spec

        self.dimensions = dimensions or self.calculate_my_dimensions()

        if self.global_colorbar:
            self.create_outer_and_subgrid()
        else:
            self.create_grid() 

        self.cells = self.make_all_cells()
    
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
        self.gs = self.figure.add_gridspec(*self.dimensions, **gs_args)
        print("just created a gridspec")
        if self.processor:
            print([division['members'] for division in self.processor.spec['divisions']])
        print(f"the figure it's attached to is {self.figure}")
        print(f"the gridspec I created is {self.gs}")
        print(f"the gridspec's id is {id(self.gs)}")
        print("")
       
    def create_outer_and_subgrid(self):
        outer_gs, main_slice, cax_slice = self.make_outer_gridspec() 
        gs_subfig = self.figure.add_subfigure(outer_gs[main_slice])
        self.figure = gs_subfig
        self.gs = gs_subfig.add_gridspec(*self.dimensions)
        cax_subfig = self.figure.add_subfigure(outer_gs[cax_slice])
        self.figure.cax = cax_subfig.add_subplot(outer_gs[cax_slice])
        self.processor.figure = self.figure
        
       
        
    def make_all_cells(self):
        return np.array([
            [self.make_cell(i, j) for j in range(self.dimensions[1])] 
            for i in range(self.dimensions[0])
        ])
    
    @property
    def no_more_processors(self):
        return self.processor and (
            self.processor.name == 'segment' or 
            self.processor.next is None
            )
        
    def make_cell(self, i, j):
        subfigure = self.make_subfigure(i, j)

        if self.no_more_processors:
            ax = subfigure.add_subplot()
            return AxWrapper(ax, subfigure, (i, j))
        else:
            return subfigure
    
    def make_subfigure(self, i, j):
        if self.colorbar_for_each_plot:
            return self.colorbar_enabled_subfigure(
                self.figure, self.gs[i, j], self.colorbar_position)
        else:
            print("I'm attaching a subfigure")
            if self.processor:
                print([division['members'] for division in self.processor.spec['divisions']])

            print(f"the figure I'm attaching to is {self.figure}")
            print(f"the gridspec I'm attaching to is {self.gs}")
            print(f"the gridspec's id is {id(self.gs)}")
        
            subfigure = self.figure.add_subfigure(self.gs[i, j])
            print(f"the new subfigure is {subfigure}\n")
            print(f"the new subfigure's position is {subfigure.bbox}")
            
            return subfigure
       
            
        
    

class AxWrapper(Base):

    def __init__(self, ax, figure, index):
        self.ax = ax  # Store the original ax
        self.figure = figure # the subfigure the ax is on
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


        
      
                        
                
                
    



