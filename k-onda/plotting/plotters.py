import os
import numpy as np
from copy import deepcopy, copy
import json


import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

from plotting.plotting_helpers import smart_title_case, PlottingMixin, format_label
from utils.utils import to_serializable, safe_get, recursive_update
from plotting.plotter_base import PlotterBase
from plotting.partition import Section, Segment, Subset
from plotting.subplotter import Figurer

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial'] 


class ExecutivePlotter(PlotterBase, PlottingMixin):
    """Makes plots, where a plot is a display of particular kind of data.  For displays of multiple 
    plots of multiple kinds of data, see the figure module."""

    def __init__(self, experiment, graph_opts=None):
        self.experiment = experiment
        self.graph_opts = graph_opts
        
    def initialize(self, calc_opts, graph_opts):
        """Both initializes values on self and sets values for the context."""
        self.calc_opts = calc_opts  
        self.graph_opts = graph_opts
        self.experiment.initialize_data()

    def plot(self, calc_opts, graph_opts, parent_figure=None, index=None):
        self.initialize(calc_opts, graph_opts)
        plot_spec = graph_opts['plot_spec']
        self.parent_figure = parent_figure
        self.process_plot_spec(plot_spec, index=index)
        if not self.parent_figure:
            self.close_plot(graph_opts.get('fname', ''))

    def process_plot_spec(self, plot_spec, index=None):

        processor_classes = {
            'section': Section,
            'segment': Segment,
            'subset': Subset
        }

        self.active_spec_type, self.active_spec = list(plot_spec.items())[0]
        processor = processor_classes[self.active_spec_type](self, index=index)
        processor.start()

    def make_fig(self):
        self.fig = Figurer().make_fig()
        self.active_fig = self.fig
            
    def close_plot(self, basename='', fig=None, do_title=True):
        
        if not fig:
            fig = self.active_fig  
        fig.delaxes(fig.axes[0])
        self.set_dir_and_filename(fig, basename, do_title=do_title)
        plt.show()
        self.save_and_close_fig(fig, basename)

    def set_dir_and_filename(self, fig, basename, do_title=True):
        tags = [basename] if basename else [self.calc_type]
        self.title = smart_title_case(' '.join([tag.replace('_', ' ') for tag in tags]))
        if do_title:
            bbox = fig.axes[0].get_position()
            fig.suptitle(self.title, fontsize=16, y=bbox.ymax + 0.1)
        self.fname = f"{'_'.join(tags)}.png"

    def save_and_close_fig(self, fig, basename):
        dirs = [self.graph_opts['graph_dir'], self.calc_type]
        path = os.path.join(*dirs)
        os.makedirs(path, exist_ok=True)
        fname = basename if basename else self.calc_type
        fig.savefig(os.path.join(path, fname), bbox_inches='tight', dpi=300)
        opts_filename = fname.replace('png', 'txt')

        with open(os.path.join(path, opts_filename), 'w') as file:
            json.dump(to_serializable(self.calc_opts), file)
        plt.close(fig)

    def delegate(self, info, is_last=False):

        def send(plot_type):
            PLOT_TYPES[plot_type]().process_calc(info, aesthetics=aesthetics, is_last=is_last)

        aesthetics = self.active_spec.get('aesthetics', {})
        if 'layers' in self.active_spec:
            for i, layer in enumerate(self.active_spec['layers']):
                if layer['attr'] == 'scatter':
                    a = 'foo'
                if i == len(self.active_spec['layers']):
                    self.active_spec['main'] = True  
                layer_aesthetics = deepcopy(aesthetics)
                aesthetics = recursive_update(layer_aesthetics, layer.get('aesthetics', {}))           
                if 'attr' in layer:
                    for row in info:
                        row['attr'] = layer['attr']
                if 'plot_type' in layer:
                    send(layer['plot_type'])
                else:
                    send(self.graph_opts['plot_type'])
        else:
            send(self.graph_opts['plot_type'])
                

class FeaturePlotter(PlotterBase, PlottingMixin):
    
    def get_aesthetic_args(self, row, aesthetics):

        aesthetic = {}
        aesthetic_spec = deepcopy(aesthetics)
        default, override, invariant = (aesthetic_spec.pop(k, {}) for k in ['default', 'override', 'invariant'])

        aesthetic.update(default)
            
        for category, members in aesthetic_spec.items():
            for member, aesthetic_vals in members.items():
                if category in row and row[category] == member:
                    recursive_update(aesthetic, aesthetic_vals)

        for combination, overrides in override.items():
            pairs = list(zip(combination.split('.')[::2], combination.split('.')[1::2]))
            if all(row.get(key, val) == val for key, val in pairs):
                aesthetic.update(overrides)

        aesthetic = recursive_update(aesthetic, invariant)

        return aesthetic
    
    def handle_broken_axes(self, data):
    # Initial data division (copying the original data)

        data_divisions = np.array(copy(data))
        if data_divisions.ndim == 1:
            data_divisions = data_divisions.reshape(1, -1)
       
            
        acks = self.active_acks
        
        # Handle data division if break_axes is present
        if hasattr(acks, 'break_axes'):
            ax_list = acks.ax_list
            for dim in acks.break_axes:
                if dim == 1:
                    data_divisions = [
                        data_divisions[slice(*arg_set)] for arg_set in acks.break_axes[1]]
                elif dim == 0:
                     data_divisions = [
                        data_divisions[:, self.get_data_slice(arg_set)] 
                        for arg_set in acks.break_axes[0]
                ]
            x_slices = acks.break_axes[0]
        else:
            length = len(data[0]) if data.ndim > 1 else len(data)
            x_slices = [(0, length*self.bin_size)]
            ax_list = [self.active_acks]
        
        return data_divisions, ax_list, x_slices
    
    def get_data_slice(self, arg_set):
        """Return a slice object for the given arg_set and bin_size."""
        return slice(*((arg_set) / self.bin_size).astype(int))
    

class HistogramPlotter(FeaturePlotter):
    
    def plot_hist(self, x, y, width, acks, aesthetic_args):
        acks.ax.bar(x, y, width=width, **aesthetic_args.get('marker', {})) 


class LinePlotter(FeaturePlotter):
    def process_calc(self, info, aesthetics=None, **_):
        attr = self.active_spec.get('attr', 'calc')
        ax = self.active_acks
        for row in info:
            val = row[attr]
            aesthetics = self.get_aesthetic_args(row, aesthetics)
            ax.plot(np.arange(len(val)), val, **aesthetics['marker'])


class WaveformPlotter(LinePlotter):
    def process_calc(self, info, aesthetics=None, is_last=False, **_):
        super().process_calc(info, aesthetics=aesthetics)
        self.label(info[0], self.active_acks, is_last)  


class CategoryPlotter(FeaturePlotter):
    
    def transform_divisions(self, divisions):
         
        new_divisions = {}
         
        for key in divisions.keys():
            if key == 'data_source':
                new_divisions[divisions['data_source']['type']] = divisions['data_source']
            else:
                new_divisions[key] = divisions[key]

        return new_divisions
    
    def assign_positions(self, divisions, base_position=0, level_names=None, prefix_labels=()):
        
        if level_names is None:
            level_names = list(divisions.keys())

        if not level_names:
            # Base case: no more divisions
            return {}

        division = level_names[0]
        remaining_levels = level_names[1:]

        division_info = divisions[division]
        spacing = division_info.get('spacing', 2)  # Get 'spacing' from division_info
        members = division_info['members']

        label_to_pos = {}
        position = base_position

        for member in members:
            current_label = prefix_labels + (member,)

            if remaining_levels:
                # Recursively assign positions for subcategories
                sub_positions = self.assign_positions(
                    divisions,
                    base_position=position,
                    level_names=remaining_levels,
                    prefix_labels=current_label
                )
                label_to_pos.update(sub_positions)

                # Update position after processing subcategories
                if sub_positions:
                    last_pos = max(sub_positions.values())
                    position = last_pos + spacing
                else:
                    # No subcategories, increment position by spacing
                    position += spacing
            else:
                # Base case: assign position to the composite label
                label_to_pos[current_label] = position
                position += spacing

        return label_to_pos
        
    def process_calc(self, info, aesthetics=None, is_last=False):
        transformed_divisions = self.active_spec['divisions']
        self.label_to_pos = self.assign_positions(transformed_divisions)
        ax = self.active_acks

        for row in info:
            composite_label = tuple(row.get(division) for division in transformed_divisions.keys())
            position = self.label_to_pos[composite_label]

            aesthetic_args = self.get_aesthetic_args(row, aesthetics)
            self.cat_width = aesthetic_args.get('cat_width', 1)
            marker_args = aesthetic_args.get('marker', {})

            self.plot_markers(position, composite_label, row, marker_args, aesthetic_args=aesthetic_args)
            self.label(row, ax, aesthetic_args, is_last)

        # Set x-ticks and labels
        positions = list(self.label_to_pos.values())
        labels = [smart_title_case(' '.join(label)) for label in self.label_to_pos.keys()]
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)

class CategoricalScatterPlotter(CategoryPlotter):

    def plot_markers(self, position, _, row, marker_args, aesthetic_args=None):
        scatter_vals = row[row['attr']]
        ax = self.active_acks
        # Generate horizontal jitter
        jitter_strength = aesthetic_args.get('max_jitter', self.cat_width/6)
        jitter = np.random.uniform(-jitter_strength, jitter_strength, size=len(scatter_vals))
        x_positions = position + jitter
        # Plot with jittered positions
        ax.scatter(x_positions, scatter_vals, **marker_args)
        # Retrieve positions and bar width
        cat_width = self.cat_width
        if 'background_color' in aesthetic_args:
            background_color, alpha = aesthetic_args.pop('background_color')
            ax.axvspan(
                position - cat_width / 2,
                position + cat_width / 2,
                facecolor=background_color, alpha=alpha)
    

class CategoricalLinePlotter(CategoryPlotter):

    def plot_markers(self, position, _, row, marker_args, aesthetic_args=None):
        ax = self.active_acks
        bar_width = self.cat_width
        divisor = aesthetic_args.get('divisor', 2)
        width = bar_width / divisor
        ax.hlines(
            row['mean'],
            position - width / 2,
            position + width / 2,
            **marker_args)

        
class BarPlotter(CategoryPlotter):
   
    def plot_markers(self, position, _, row, marker_args, aesthetic_args=None):
        self.active_acks.bar(position, row[row['attr']], **marker_args)
        

class PeriStimulusPlotter(FeaturePlotter):
    
    def __init__(self ):
        self.base = self.calc_opts.get('base', 'event') 
        self.pre, self.post = (getattr(self, f"{opt}_{self.base}") for opt in ('pre', 'post'))
        self.marker_names = []
        
    def process_calc(self, info, aesthetics=None, is_last=False, **_):
                
        for row in info:
            attr = self.active_spec.get('attr', 'calc')
            data = row[attr]
            aesthetic_args = self.get_aesthetic_args(row, aesthetics)
            data_divisions, ax_list, x_slices = self.handle_broken_axes(data)
            
            for i, (ax, data, x_slice) in enumerate(zip(ax_list, data_divisions, x_slices)):
                self.plot_row(ax, data, row, i, aesthetic_args, data_source=row['data_source'])
                #self.set_x_ticks(ax, data, x_slice)
                self.place_indicator(ax, aesthetic_args)
                self.label(row, ax, aesthetic_args, is_last)   
                
    def set_x_ticks(self, ax, data, x_slice):

        length = len(data[0]) if data.ndim > 1 else len(data)

        nearest_power_of_ten = 10 ** np.floor(np.log10(self.pre + self.post))
        rounded_step = nearest_power_of_ten 
        if 0 < self.pre < rounded_step:
            beginning = 0
        else:
            beginning = -self.pre
        
        manual_ticks = np.arange(beginning, self.post, step=rounded_step) 
        ax.set_xticks(manual_ticks)

        # Get the existing tick positions (in bins)
        existing_ticks = ax.get_xticks()

        # Filter the ticks to only those within the visible range of the data (i.e., corresponding to x_slice)
        visible_ticks = [tick for tick in existing_ticks if 0 <= tick <= length]

        # Calculate the start and end time for the current x_slice (in seconds)
        x_slice_start_time = x_slice[0] - self.pre  
        x_slice_end_time = x_slice[1] - self.pre

        # Create a time range that matches the visible ticks, from x_slice_start_time to x_slice_end_time
        tick_range = np.linspace(x_slice_start_time, x_slice_end_time, len(visible_ticks))

        # Set the x-tick positions and labels
        ax.set_xticks(visible_ticks)  # Use only the visible ticks
        ax.set_xticklabels([f"{label:.2f}" for label in tick_range])  # Labels in seconds
            

    def place_indicator(self, ax, aesthetic_args):
        indicator = aesthetic_args.get('indicator', {})
        if not indicator:
            return
        indicator_type = indicator.get('type')
        when = indicator['when']

        if indicator_type == 'patch':
            if len(when) == 2:
                width = when[1] - when[0]
                # Create a transformation: x in data coords, y in axes coords
                transform = transforms.blended_transform_factory(ax.transData, ax.transAxes)
                ax.add_patch(plt.Rectangle(
                    (when[0], 0), width, 1,
                    facecolor='gray', alpha=0.3,
                    transform=transform  # Apply the transformation
                ))
                

class RasterPlotter(PeriStimulusPlotter):
    
    def process_calc(self, *args, **kwargs):
        super().process_calc(*args, **kwargs)
                
    def plot_row(self, ax, data, row, slice_no, aesthetic_args, data_source=None):
        ax.set_ylim(0, len(data))  # Set ylim based on the number of rows
        line_length = aesthetic_args.get('line_length', .9)   
        marker_args = aesthetic_args.get('marker', {})  
        row_label_spec = safe_get(aesthetic_args, ['label', 'ax', 'row_labels'])
        if data_source and row_label_spec and slice_no == 0:
            if isinstance(row_label_spec, str) and row_label_spec.startswith('lambda'):
                row_label_func = eval(row_label_spec)
                row_labels = row_label_func(row['data_source'])
            else:
                row_labels = format_label(row_label_spec, data_source)
        else:
            row_labels = ['' for _ in range(len(data))]

        # Plot spikes on the vlines
        for i, spiketrain in enumerate(data):
            ax.text(-3, i + line_length / 2, row_labels[i], ha='right', va='center', rotation=45, fontsize=8)  # Adjust x-coordinate to position label
            for j, spike in enumerate(spiketrain):
                if spike:
                    ax.vlines(j, i, i + line_length, **marker_args)
    
                
class PeriStimulusHistogramPlotter(PeriStimulusPlotter, HistogramPlotter):
        
    def plot_row(self, ax, data, row, slice_no, aesthetic_args, **_):
        if data.ndim > 1:
            data = data[0]
        x = np.linspace(-self.pre, self.post, len(data)+1)[:-1]
        y = data
        self.plot_hist(x, y, self.calc_opts['bin_size'], ax, aesthetic_args)
        

PLOT_TYPES = {'categorical_scatter': CategoricalScatterPlotter,
              'line_plot': LinePlotter,
              'bar_plot': BarPlotter,
              'waveform': WaveformPlotter,
              'categorical_line': CategoricalLinePlotter,
              'raster': RasterPlotter,
              'psth': PeriStimulusHistogramPlotter}  
