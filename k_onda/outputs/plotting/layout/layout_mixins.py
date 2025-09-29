import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba

from collections import defaultdict

from k_onda.utils import safe_get


class AxShareMixin:

    def set_share_bins(self):
        if self.spec is not None:
            self.shared_axes_spec = safe_get(self.spec, ['aesthetics', 'ax', 'share'])
        if self.shared_axes_spec:
            self.share_bins = {'x': [], 'y': []}

    def finalize_shared_axes(self):
        for child in self.children:
            child.finalize_shared_axes()
        
        if not self.share_bins:
            return
        
        for axis_key in ('x', 'y'):
            groups = self.group(self.share_bins, axis_key)
            for _, axes in groups.items():
                self._normalize_then_share(axes, axis_key, self.shared_axes_spec or {})

    def group(self, bins, axis_key):
        spec = self.shared_axes_spec
        dim = spec[axis_key] if isinstance(spec, dict) else None

        for i, m in enumerate(('row', 'col')):
            if m == dim:
                groups = defaultdict(list)
                for ax in bins[axis_key]:
                    groups[ax.index[i]].append(ax)
                return groups
            
        return {'all': bins[axis_key]}
    
    def _apply_scale(self, ax, axis_key, spec_val, anchor_scale):
        set_scale = getattr(ax.obj, f"set_{axis_key}scale")
        if isinstance(spec_val, dict):
            scale = spec_val.get('scale', anchor_scale or 'linear')
            kw = {k: v for k, v in spec_val.items() if k != 'scale'}
            set_scale(scale, **kw)
        elif isinstance(spec_val, str):
            set_scale(spec_val)
        else:
            # No explicit spec → copy anchor
            set_scale(anchor_scale)

    def _harmonize_inversion(self, ax, axis_key, anchor_inverted):
        is_inv = getattr(ax.obj, f"{axis_key}axis_inverted")()
        if is_inv != anchor_inverted:
            getattr(ax.obj, f"invert_{axis_key}axis")()

    def _normalize_then_share(self, axes, axis_key, share_spec):
        if len(axes) < 2:
            return
        first, *rest = axes

        # 1) choose canonical scale
        anchor_scale = getattr(first.obj, f"get_{axis_key}scale")()
        spec_val = None
        if isinstance(share_spec, dict) and axis_key in share_spec:
            spec_val = share_spec[axis_key]

        # 2) normalize everyone’s scale to canonical
        #    (either explicit spec, or anchor’s scale)
        for ax in [first, *rest]:
           self._apply_scale(ax, axis_key, spec_val, anchor_scale)

        # 3) match inversion state (left/right or up/down)
        anchor_inv = getattr(first.obj, f"{axis_key}axis_inverted")()
        for ax in rest:
            self._harmonize_inversion(ax, axis_key, anchor_inv)

        # 4) now it’s safe to share
        for ax in rest:
            getattr(ax.obj, f"share{axis_key}")(first.obj)


class LegendMixin:

    # TODO: fix this.  It is broken.  The fundamental problem is: how does the user 
    # specify in which cell they want the legend to fall?  How does an individual layout
    # object have knowledge about whether it contains that cell?  More complicated than it sounds,
    # must be thought about when I have mental energy, which is not right now.

    def record_entry_for_legend(self, entry, legend, cells_with_legend):
        entry['handle'] = self.get_handle(entry)
        entry['label'] = self.get_entry_label(entry, legend)
        index_key = ','.join(str(d) for d in entry['index'])
        if index_key in legend or 'all' in legend:
            cells_with_legend.append((entry['cell'], index_key))

    def make_legend(self, cells_with_legend, legend):
        
        unique = {}
        for e in self.info:
            h, label = e['handle'], e['label']
            key = (label, self.artist_style_key(h))  # or just artist_style_key(h)
            if key not in unique:
                unique[key] = (h, label)

        if unique:
            handles, labels = zip(*unique.values())
        else:
            handles, labels = [], []
        
        for ax, index_key in cells_with_legend:
            legend_key = legend.get(index_key, {})
            legend_key['loc'] = legend_key.get('loc') or 'lower center'
            legend_key['bbox_to_anchor'] = legend_key.get('bbox_to_anchor') or (.8, .9)
            ax.legend(handles, labels, **legend_key)

    def get_handle(self, entry):
        if self.plot_type in ['psth', 'bar_plot']:
            return self.make_bar_handles_from_entry(entry)
        if self.plot_type in ['categorical_line', 'vertical_line', 'line_plot']:
            # todo: is this just getting written over by get_handle on LinePlotter?
            return self.make_line_handles_from_entry(entry)
        else:
            raise NotImplemented("can only make legends from handles for line and bar graphs")
        

    def make_line_handles_from_entry(self, entry):
   
        aesthetic_args = self.get_aesthetic_args(entry)
        marker = aesthetic_args.get('marker', {})
        
        color = marker.get('color', 'CO')
        linestyle = marker.get('linestyle', '-')  # solid line by default
        linewidth = marker.get('linewidth', 2)
        markerstyle = marker.get('marker', None)
        markersize = marker.get('markersize', 6)
        alpha = marker.get('alpha', None)

        line = Line2D([0], [0],
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    marker=markerstyle,
                    markersize=markersize,
                    alpha=alpha)
      
        return line  

    def make_bar_handles_from_entry(self, entry):
    
        aesthetic_args = self.get_aesthetic_args(entry)
        marker = aesthetic_args.get('marker', {})
        color = marker.get('color', 'CO')
        edgecolor = marker.get('edgecolor', None)
        hatch = marker.get('hatch', None)
        alpha = marker.get('alpha', None)

        patch = mpatches.Patch(facecolor=color, edgecolor=edgecolor,
                            hatch=hatch, alpha=alpha)
       
        return patch
    
    @staticmethod
    def _rgba(x):
        r, g, b, a = to_rgba(x)  # normalizes names/tuples/etc.
        return (round(r, 4), round(g, 4), round(b, 4), round(a, 4))

    def artist_style_key(self, artist):
        """Return a hashable key that captures how the artist looks in the legend."""
        if isinstance(artist, mpatches.Patch):
            face = self._rgba(artist.get_facecolor()) if artist.get_fill() else None
            edge = self._rgba(artist.get_edgecolor()) if artist.get_edgecolor() is not None else None
            return (
                'patch',
                face,
                edge,
                round(artist.get_linewidth() or 0, 4),
                artist.get_linestyle(),
                artist.get_hatch() or '',
                round(artist.get_alpha(), 4) if artist.get_alpha() is not None else None,
                tuple(type(pe).__name__ for pe in artist.get_path_effects()),
            )
        if isinstance(artist, Line2D):
            # In case you also dedupe lines
            return (
                'line',
                self._rgba(artist.get_color()),
                round(artist.get_linewidth() or 0, 4),
                artist.get_linestyle(),
                artist.get_marker(),
                round(artist.get_markersize() or 0, 4),
                None if artist.get_markerfacecolor() == 'auto' else self._rgba(artist.get_markerfacecolor()),
                None if artist.get_markeredgecolor() == 'auto' else self._rgba(artist.get_markeredgecolor()),
            )
        # Fallback: class name only (adjust if you use other artist types)
        return (artist.__class__.__name__,)

            
    def calculate_legend_y_position(self, entries):
        # Collect all y-data from the entries based on the attribute indicated in each entry.
        all_y_data = []
        for entry in entries:
            attr = entry.get('attr')
            if attr and attr in entry:
                data_points = entry[attr]
                # If data_points is an iterable (like a list or array), extend the list
                try:
                    iter(data_points)
                    all_y_data.extend(data_points)
                except TypeError:
                    all_y_data.append(data_points)

        # Compute a safe anchor point for the legend.
        if all_y_data:
            max_y = max(all_y_data)
            min_y = min(all_y_data)
            y_range = max_y - min_y
            # Increase the y coordinate by 5% of the range above the top data point.
            anchor_y = max_y + 0.05 * y_range
        else:
            anchor_y = .9  # default anchor if no data is found
        return anchor_y


class ColorbarMixin:

    @property
    def has_colorbar(self):
        return bool(self.colorbar_spec)

    @property
    def colorbar_spec(self):
        return self.spec and self.spec.get('legend', {}).get('colorbar', {})
           
    @property
    def colorbar_for_each_plot(self):
        return self.has_colorbar and self.colorbar_spec.get('share') in ['each', None]

    @property
    def global_colorbar(self):
        return self.has_colorbar and self.colorbar_spec.get('share') == 'global'

    @property
    def colorbar_position(self):
        return self.colorbar_spec.get('position')
    
    def create_outer_and_subgrid(self):
        outer_grid, main_slice, cax_slice = self.make_outer_grid() 
        self.figure = outer_grid[main_slice]
        self.figure.cax = outer_grid[cax_slice].add_subplot(111)
        self.processor.figure = self.figure
    
    def make_outer_grid(self):
       
        i = ['top', 'bottom', 'left', 'right'].index(self.colorbar_position)
        outer_dim = [1, 1]
        outer_dim[i//2] += 2
        ratio_string = ['height', 'width'][i//2]
        ratios = [1, .025, .075]
        if not i % 2:
            ratios.reverse()
        outer_grid = self.subfigures(*outer_dim, **{f'{ratio_string}_ratios': ratios})
        location_of_main_figure = [0, 0]
        location_of_colorbar_ax = [0, 0]

        location_of_colorbar_ax[i//2] += 1
        if not i%2:
            location_of_main_figure[i//2] += 2

        return (outer_grid, tuple(location_of_main_figure), 
                tuple(location_of_colorbar_ax))
    


        





    

        



