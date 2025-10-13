from collections import defaultdict
from copy import deepcopy

from matplotlib.gridspec import  GridSpec, GridSpecFromSubplotSpec
import numpy as np

from k_onda.core import Base
from .layout_mixins import ColorbarMixin, AxShareMixin


class Layout(Base, ColorbarMixin, AxShareMixin):

    def __init__(self, parent, index, figure=None, processor=None, dimensions=None,
                 gs_args=None):
        
        self.parent = parent
        if self.parent is not None:
            self.parent.children.append(self)
        self.children = []

        self.index = index
        self.figure = figure
        self.processor = processor
        
        self.gs_args = gs_args if gs_args else {}

        if self.processor is None:
            self.processor_type = None
            self.spec = None
        else:
            self.processor_type = processor.name
            self.spec = self.processor.spec

        self.shared_axes_spec = None

        self._dimensions = dimensions 
        
        self.share_bins = None 
        self.set_shared_axes_spec()

        self.create_grid()
       
        if self.processor and self.processor.spec_type == 'segment':
            if hasattr(self.parent, 'label_figure'):
                self.label_figure = self.parent.label_figure
            else:
                self.label_figure = self.figure

        if self.no_more_processors:
            self.cells = self.make_all_axes()
            
        else:
            self.cells = self.subfigure_grid

    def __repr__(self):
        try:
            dims = getattr(self, "dimensions", None)
            proc = getattr(self, "processor_type", None)
            spec = getattr(self, "spec", None)

            if isinstance(spec, dict) and "divisions" in spec:
                try:
                    divisions = " ".join(d["divider_type"] for d in spec["divisions"])
                except Exception:
                    divisions = None
            else:
                divisions = None

            return f"Layout Object {dims} {proc} {divisions}"
        except Exception as e:
            return f"<Layout index={getattr(self,'index',None)} repr_error={e!r}>"
        
    
    @property
    def dimensions(self):
        if self._dimensions is None:
            self._dimensions = self.calculate_my_dimensions()
        return self._dimensions


    def root(self):
        if self.parent is None:
            return self
        else:
            return self.parent.root()

    def subfigures(self, nrows, ncols, **kwargs):
        """
        Create a (nrows x ncols) array of SubFigure wrappers.

        If `wspace` or `hspace` are supplied, the function inserts blank
        sub-figures between the real ones instead of relying on
        `subplots_adjust`, so you get predictable padding that behaves just
        like extra columns/rows.

        Parameters
        ----------
        nrows, ncols : int
            Logical grid size (counts *only* the real sub-figures).
        wspace, hspace : float, optional
            Relative width/height of the blank padding columns/rows
            compared to a real sub-figure (default 0 = no padding).
        **kwargs
            Forwarded to `Figure.subfigures` after `wspace`/`hspace`
            are popped.

        Returns
        -------
        ndarray[(nrows, ncols)]
            Array of `SubfigWrapper`s for the real sub-figures.
        """
        # -- pull the “spacing” options out of kwargs ---------------------
        wspace = kwargs.pop("wspace", 0)
        hspace = kwargs.pop("hspace", 0)

        # If no extra spacing, fall back to the old behaviour -------------
        if not wspace and not hspace:
            raw = self.figure.subfigures(nrows, ncols, **kwargs)
            if not isinstance(raw, (list, np.ndarray)):
                return np.full((nrows, ncols),
                            SubfigWrapper(raw, (0, 0), self))
            raw = np.asarray(raw)
            if raw.ndim == 1:                      # 1-D ⇒ reshape to 2-D
                raw = raw.reshape(nrows, ncols)
            return np.array([[SubfigWrapper(sf, (i, j), self)
                            for j, sf in enumerate(row)]
                            for i, row in enumerate(raw)])

        # -- build the “big” grid with padding sub-figures ---------------
        padded_rows = nrows + (nrows - 1) if hspace else nrows
        padded_cols = ncols + (ncols - 1) if wspace else ncols

        # width / height ratios: [real, pad, real, pad, …]
        width_ratios = []
        for c in range(ncols):
            width_ratios.append(1)
            if c < ncols - 1 and wspace:
                width_ratios.append(wspace)
        height_ratios = []
        for r in range(nrows):
            height_ratios.append(1)
            if r < nrows - 1 and hspace:
                height_ratios.append(hspace)

        kwargs.setdefault("width_ratios", width_ratios)
        kwargs.setdefault("height_ratios", height_ratios)

        big = self.figure.subfigures(padded_rows, padded_cols, **kwargs)
        big = np.asarray(big).reshape(padded_rows, padded_cols)

        # -- pick out only the “real” slots (even indices) ---------------
        row_idx = range(0, padded_rows, 2 if hspace else 1)
        col_idx = range(0, padded_cols, 2 if wspace else 1)

        subfig_grid = np.empty((nrows, ncols), dtype=object)
        for i, rr in enumerate(row_idx):
            for j, cc in enumerate(col_idx):
                subfig_grid[i, j] = SubfigWrapper(big[rr, cc], (i, j), self)

        return subfig_grid
            
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

        subfigure_args = self.spec.get('figure', {}) if self.processor else {}
        subplots_adjust = subfigure_args.pop('subplots_adjust', None)
        if subplots_adjust:
            self.figure.get_figure().subplots_adjust(**subplots_adjust)
            
        self.subfigure_grid = self.subfigures(*self.dimensions, **self.gs_args, 
                                              **subfigure_args)
        
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
        
    def make_all_axes(self):
        if self.spec.get('break_axis'):
            creation_func = lambda i, j: BrokenAxes(self.subfigure_grid[i, j], self, (i, j), self.spec['break_axis'])
        else:
            creation_func = lambda i, j: AxWrapper(self.subfigure_grid[i, j].subfigures(1, 1).add_subplot(),
                                                   (i, j), self)
        
        cells = np.array([
            [creation_func(i, j) 
             for j in range(self.dimensions[1])] 
             for i in range(self.dimensions[0])
        ])

        return cells
    
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
        return self.is_extreme_index(self.index, self.layout, dim, last)
    
    def is_in_extreme_position(self, axis, last, absolute):
        dim = int(not axis == 'x')

        is_extreme_within_parent = self.is_extreme_within_parent(dim, last)
        
        if not absolute:
            return is_extreme_within_parent

        return is_extreme_within_parent and self.parent_is_extreme(dim, last)


class SubfigWrapper(Wrapper):

    name = 'subfig'

    def __init__(self, subfig, index, layout):
        super().__init__(subfig, index, layout)
        self.subfig = subfig


class AxWrapper(Wrapper):

    name = 'ax'

    def __init__(self, ax, index, layout):
        super().__init__(ax, index, layout)
        self.ax = ax
    

class BrokenAxes(Base):

    name = 'broken_axes'
    
    def __init__(self, fig, parent_layout, index, break_axes, aspect=None):
        self.break_axes = break_axes
        self.index = index
        self.fig = fig
        self.layout = parent_layout
        self.aspect = aspect
        
        self.dim0_breaks = len(self.break_axes.get(1, {}).get('splits', [])) or 1
        self.dim1_breaks = len(self.break_axes.get(0, {}).get('splits', [])) or 1
        
        self.subfig = SubfigWrapper(self.fig, index, self.layout)
        self.gs = GridSpec(self.dim0_breaks, self.dim1_breaks, figure=self.subfig.obj)

        self.axes, self.ax_list = self._create_subplots()

        self._share_axes_and_hide_spines()

    def __getattr__(self, name):
        # Forward any unknown attribute access to the subfigure
        return getattr(self.subfig, name)

    def _create_subplots(self):
        axes = []
        ax_list = []
        for i0 in range(self.dim0_breaks):
            row = []
            for j1 in range(self.dim1_breaks):
                gridspec_slice = self.gs[i0, j1]
                ax = self.subfig.add_subplot(gridspec_slice)
                if self.aspect:
                    ax.set_box_aspect(self.aspect)
                ax_wrapper = AxWrapper(ax, (i0, j1), self.layout)
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
                    getattr(ax, f"share{dim_num}")(first.obj)

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


        
      
                        
                
                
    



