import numpy as np
from plotter_base import PlotterBase
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from copy import copy


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
        self._dimensions_in_axes = dimensions
        self._dimensions_in_acks = None
        self.grid_keywords = grid_keywords
        self.frame_ax = self.create_invisible_frame()  # Generalize the invisible frame creation
        self.gs = self.create_grid()  # Create the gridspec for the actual data
        self.invisible_axes = invisible_axes or []
        self._axes = None
        if first:
            self._ax_visibility = np.array([[None]])
        else:
            self._ax_visibility = np.array(
                [[True if (i, j) not in self.invisible_axes else False 
                  for j in range(self.dimensions_in_axes[1])] 
                  for i in range(self.dimensions_in_axes[0])])
            self.adjust_gridspec_bounds() # TODO: I need a version of this that works with the frame ax
            # and both should take arguments, effectively giving both the frame and the data independently
            # operable left, right, top, and bottom parameters
    
    @property
    def axes(self):
        if self._axes is None:
            self._axes = self.make_all_axes()
        return self._axes
    
    @property
    def ax_list(self):
        return [a for r in self.axes for a in r]
    
    @property
    def dimensions_in_axes(self):
        if self._dimensions_in_axes is None:
            self.calculate_my_dimensions()
        return self._dimensions_in_axes
    
    @property
    def dimensions_in_acks(self):
        if self._dimensions_in_acks is None:
            self.calculate_my_dimensions()
        return self._dimensions_in_acks
    
    @property
    def ax_visibility(self):
        return self._ax_visibility
    
    @ax_visibility.setter
    def ax_visibility(self, visibility):
        self._ax_visibility = visibility

    def calculate_my_dimensions(self):
        dims = [1, 1]
        if self.first:
            self._dimensions_in_axes = dims
            self._dimensions_in_acks = dims
            return
        for division in self.spec['divisions'].values():
            if 'dim' in division:
                dims[division['dim']] = len(division['members'])
        self._dimensions_in_acks = copy(dims)
        for i, breaks in self.active_spec.get('break_axis', {}).items():
            dims[int(not i)] *= len(breaks)
        self._dimensions_in_axes = copy(dims)

    def create_frame_grid(self):
        """Create a 1x1 gridspec for the invisible frame to surround the data."""
        parent_gridspec = GridSpecFromSubplotSpec(
            1, 1,  # A single cell for the invisible frame
            subplot_spec=self.parent.gs[*self.index]  # This becomes the parent gridspec
        )
        return parent_gridspec

    def create_grid(self):
        """Create the gridspec for the actual data, nested inside the invisible frame."""
        frame_gridspec = self.create_frame_grid()

        # Create the gridspec for the data inside the invisible frame
        data_gridspec = GridSpecFromSubplotSpec(
            *self.dimensions_in_acks,  # The number of rows and columns for the actual data
            subplot_spec=frame_gridspec[0]  # Use the only cell in the frame for the data
        )

        return data_gridspec
    
    def adjust_gridspec_bounds(self):
        # Adjust the position of the axes manually to create padding
        for i, row in enumerate(self.get_ax_wrappers()):
            for j, ax in enumerate(row):
                pos = ax.get_position()  # Get the current position of the axis
                # Adjust the position to make the grid take up less space
                pos.x0 += 0.005  # Shrink from the left
                pos.x1 -= 0.005  # Shrink from the right
                pos.y0 += 0.005  # Shrink from the bottom
                pos.y1 -= 0.005  # Shrink from the top
                ax.set_position(pos)  # Apply the new position

        self.fig.canvas.draw_idle()  # Update the figure
        
    def create_invisible_frame(self):
        """Creates an invisible frame around the specified slice of the gridspec."""
        gridspec_slice = self.parent.gs[*self.index]  # This selects the slice (e.g., 2x2 corner)

        # Create the axis but hide all the spines and ticks
        frame_ax = self.fig.add_subplot(gridspec_slice, frame_on=True)
        
        # Make spines invisible but keep the axis for labeling
        for spine in ['top', 'bottom', 'left', 'right']:
            frame_ax.spines[spine].set_visible(False)

        # Hide ticks but keep the labels visible
        frame_ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        return frame_ax  # Return the axis (which will still render labels)
        
    def make_all_axes(self):
        return np.array([
            [self.make_acks(i, j) for j in range(self.dimensions_in_acks[1])] 
            for i in range(self.dimensions_in_acks[0])
        ])
        
    def make_acks(self, i, j):
        break_axes = self.active_spec.get('break_axis', {}) if not self.first else {}

        # Check if there are any breaks
        if break_axes:
            return BrokenAxes(
                fig=self.fig, 
                parent_gridspec=self.gs, 
                index=(i, j), 
                break_axes=break_axes, 
                aspect=self.aspect
            )
        else:
            gridspec_slice = self.gs[i, j]
            ax = self.fig.add_subplot(gridspec_slice)
            ax.set_visible(self.ax_visibility[i, j])
            if self.aspect:
                ax.set_box_aspect(self.aspect)
            return AxWrapper(ax, (i, j))
    
    def mark_edges_of_component(self, xy):
        for ax in self.axes.flatten():
            if ax.index[0] == xy[0][1] - 1: # the bottommost row
                ax.bottom_edge = True
            if ax.index[1] == xy[1][0]: # the leftmost column
                ax.left_edge = True

    def apply_aesthetics(self, aesthetics):
        for key, val in aesthetics.get('ax', {}).get('border', {}).items():
            spine, tick, label = (val[i] in ['T', True, 'True'] for i in range(3))
            for ax in self.active_acks.ax_list:
                ax.spines[key].set_visible(spine)
                ax.tick_params(**{f"label{key}":label, key:tick})
        self.apply_shared_axes(aesthetics)
        
    def apply_shared_axes(self, aesthetics):
        share = aesthetics.get('ax', {}).get('share', [])
        rows_of_axes = self.get_ax_wrappers()
        columns_of_axes = list(zip(*rows_of_axes))  # Transpose to get columns

        if 'y' in share:
            for row in rows_of_axes:
                first_ax = row[0].ax
                for acks in row[1:]:
                    acks.ax.sharey(first_ax)
                    acks.ax.tick_params(labelleft=False)

        for col in columns_of_axes:
            last_ax = col[-1].ax
            for acks in col[0:-1]:
                acks.ax.sharex(last_ax)
                acks.ax.tick_params(labelbottom=False)

    def get_ax_wrappers(self):
        """Return a 2D array of AxWrapper objects from axes, expanding BrokenAxis if necessary."""
        ax_wrapper_grid = []

        for row in self.axes:
            ax_wrapper_row = []
            for ax in row:
                if isinstance(ax, BrokenAxes):
                    # If it's a BrokenAxis, extend with its internal AxWrapper objects
                    for broken_row in ax.axes:
                        ax_wrapper_row.extend(broken_row)
                else:
                    # If it's an AxWrapper, append directly
                    ax_wrapper_row.append(ax)
            ax_wrapper_grid.append(ax_wrapper_row)

        return ax_wrapper_grid


class AxWrapper(PlotterBase):

    def __init__(self, ax, index):
        self.ax = ax  # Store the original ax
        self.ax_list = [self]
        self.index = index
        self.bottom_edge = None
        self.left_edge = None

    def __getattr__(self, name):
        # Forward any unknown attribute access to the original ax
        return getattr(self.ax, name)
    

class BrokenAxes(PlotterBase):
    
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


        
      
                        
                
                
    



