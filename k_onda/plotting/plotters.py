import os
from copy import deepcopy, copy
import json
import uuid

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np

from .plotting_helpers import smart_title_case, PlottingMixin, format_label
from .plotter_base import PlotterBase
from .partition_simplified import Section, Segment, Series, Container, ProcessorConfig
from .layout_simplified import Figurer
from k_onda.utils import to_serializable, safe_get, recursive_update, PrepMethods

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial'] 


class ExecutivePlotter(PlotterBase, PlottingMixin, PrepMethods):
    """Makes plots, where a plot is a display of particular kind of data.  For displays of multiple 
    plots of multiple kinds of data, see the figure module."""

    def __init__(self, experiment):
        self.experiment = experiment
        self.write_opts = None
              
    def plot(self, opts, index=None):
        if 'calc_opts' in opts:
            self.calc_opts = opts['calc_opts']
            self.experiment.initialize_data()
        self.write_opts = opts.get('write_opts', {}) # TODO: add user formatted filename
        plot_spec = opts['plot_spec']
        self.process_plot_spec(plot_spec)
        self.close_plot(opts.get('fname', ''))

    def process_plot_spec(self, plot_spec):

        processor_classes = {
            'section': Section,
            'segment': Segment,
            'series': Series,
            'container': Container
        }
    
        config = ProcessorConfig(self, plot_spec, figure=self.make_fig(), index=[0, 0])
        processor = processor_classes[config.spec_type](config)
        processor.start()

    def make_fig(self):
        self.fig = Figurer().make_fig()
        self.active_fig = self.fig
        return self.fig
    
    def construct_path(self):
        # Fill fields
        root, fname, path = [
            self.fill_fields(self.write_opts.get(key)) 
            for key in ['root', 'fname', 'path']
        ]
        
        # If user explicitly set 'path', we skip building our own path
        if path:
            self.file_path = path
    
        else:
            # Fallback to some default root
            if not root:
                root = self.experiment.exp_info.get('data_path', os.getcwd())
            
            # Fallback to some default fname
            if not fname:
                fname = '_'.join([self.kind_of_data, self.calc_type])
            
            self.title = smart_title_case(fname.replace('_', ' '))
            
            # Build partial path (no extension yet)
            self.file_path = os.path.join(root, self.kind_of_data, fname)

        self.handle_collisions()
        
        # Build final file path and an opts file
        ext = self.write_opts.get('extension', '.png')
        self.opts_file_path = self.file_path + '.txt'
        self.file_path += ext

    def handle_collisions(self):
        if os.path.exists(self.file_path) and not self.write_opts.get('allow_overwrite', True):
            uuid_str = str(uuid.uuid4())[:self.write_opts.get('unique_hash', 8)]
            basename = os.path.basename(self.file_path)
            dir_ = os.path.dirname(self.file_path)
            new_name = f"{basename}_{uuid_str}"
            self.file_path = os.path.join(dir_, new_name)

    def close_plot(self, basename='', fig=None, do_title=True):
        
        if not fig:
            fig = self.active_fig  
        #fig.delaxes(fig.axes[0])
        plt.show()
        self.save_and_close_fig(fig, basename)
       
    def save_and_close_fig(self, fig, do_title=True):
        
        self.construct_path()

        if do_title:
            bbox = fig.axes[0].get_position()
            fig.suptitle(self.title, fontsize=16, y=bbox.ymax + 0.1)
       
        fig.savefig(self.file_path, bbox_inches='tight', dpi=300)

        with open(self.opts_file_path, 'w') as file:
            json.dump(to_serializable(self.calc_opts), file)

        plt.close(fig)
        self.active_fig = None

    def delegate(self, cell, info=None, spec=None, plot_type=None, is_last=False):

        def send(plot_type):
            PLOT_TYPES[plot_type]().process_calc(selected_info, spec, cell, aesthetics=aesthetics, is_last=is_last)

        base_aesthetics = spec.get('aesthetics', {})
        if 'layers' in spec:
            for i, layer in enumerate(spec['layers']):
                if i == len(spec['layers']):
                    spec['main'] = True 
                selected_info = [row for row in info if row['layer'] == i]
                layer_aesthetics = deepcopy(base_aesthetics)
                aesthetics = recursive_update(layer_aesthetics, layer.get('aesthetics', {}))           
                if 'plot_type' in layer:
                    send(layer['plot_type'])
                else:
                    send(plot_type)
        else:
            selected_info = info
            aesthetics = base_aesthetics
            send(plot_type)
                

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
       
            
        acks = self.active_cell
        
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
            ax_list = [self.active_cell]
        
        return data_divisions, ax_list, x_slices
    
    def get_data_slice(self, arg_set):
        """Return a slice object for the given arg_set and bin_size."""
        return slice(*((arg_set) / self.bin_size).astype(int))
    

class HistogramPlotter(FeaturePlotter):
    
    def plot_hist(self, x, y, width, acks, aesthetic_args):
        acks.ax.bar(x, y, width=width, **aesthetic_args.get('marker', {})) 


class LinePlotter(FeaturePlotter):
    def process_calc(self, info, spec, aesthetics=None, **_):
        attr = spec.get('attr', 'calc')
        ax = self.active_cell
        for row in info:
            val = row[attr]
            aesthetics = self.get_aesthetic_args(row, aesthetics)
            ax.plot(np.arange(len(val)), val, **aesthetics['marker'])


class WaveformPlotter(LinePlotter):
    def process_calc(self, info, aesthetics=None, is_last=False, **_):
        super().process_calc(info, aesthetics=aesthetics)
        self.label(info[0], self.active_cell, is_last)  


class CategoryPlotter(FeaturePlotter):
    
    def transform_divisions(self, divisions):
         
        new_divisions = {}
         
        for key in divisions.keys():
            if key == 'data_source':
                new_divisions[divisions['data_source']['type']] = divisions['data_source']
            else:
                new_divisions[key] = divisions[key]

        return new_divisions
    
    def assign_positions(self, divisions, aesthetics, base_position=0, label_prefix=None):
        
        if not label_prefix:
            label_prefix = []


        if not divisions:
            # Base case: no more divisions
            return {}
        
        division = divisions[0]
        remaining_divisions = divisions[1:]

        spacing = aesthetics.get('default', {}).get('spacing', 2)  # Get 'spacing' from division_info
        members = division['members']

        label_to_pos = {}
        position = base_position

        for member in members:
            current_label = deepcopy(label_prefix)
            if isinstance(member, str):
                current_label.append(member)
            else:
                current_label.extend(list(v for v in member.values()))
  
            if remaining_divisions:
                # Recursively assign positions for subcategories
                sub_positions = self.assign_positions(
                    remaining_divisions,
                    aesthetics,
                    base_position=position,
                    label_prefix=current_label
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
                label_to_pos[tuple(current_label)] = position
                position += spacing

        return label_to_pos
    
    def get_composite_label(self, spec, row):
        label = []
        divider_types = set([division['divider_type'] for division in spec['divisions']])
        for divider_type in divider_types:
            if isinstance(row[divider_type], str):
                label.append(row[divider_type])
            else:
                label.extend([v for d in row[divider_type] for v in d.values()])
        return tuple(label)
       
        
    def process_calc(self, info, spec, ax, aesthetics=None, is_last=False):
        transformed_divisions = deepcopy(spec['divisions'])
        self.label_to_pos = self.assign_positions(transformed_divisions, aesthetics)

        for row in info:
            composite_label = self.get_composite_label(spec, row)
            position = self.label_to_pos[composite_label]

            aesthetic_args = self.get_aesthetic_args(row, aesthetics)
            self.cat_width = aesthetic_args.get('cat_width', 1)
            marker_args = aesthetic_args.get('marker', {})

            self.plot_markers(ax, position, composite_label, row, marker_args, aesthetic_args=aesthetic_args)
            self.label(row, ax, aesthetic_args, is_last)

        # Set x-ticks and labels
        positions = list(self.label_to_pos.values())
        labels = [smart_title_case(' '.join(label)) for label in self.label_to_pos.keys()]
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)

class CategoricalScatterPlotter(CategoryPlotter):

    def plot_markers(self, ax, position, _, row, marker_args, aesthetic_args=None):
        scatter_vals = row[row['attr']]

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
                position - cat_width / 4, # TODO cat_width isn't really doing anything now
                position + cat_width / 4,
                facecolor=background_color, alpha=alpha)
    

class CategoricalLinePlotter(CategoryPlotter):

    def plot_markers(self, ax, position, _, row, marker_args, aesthetic_args=None):
        bar_width = self.cat_width
        divisor = aesthetic_args.get('divisor', 2)
        width = bar_width / divisor
        ax.hlines(
            row['mean'],
            position - width / 2,
            position + width / 2,
            **marker_args)

        
class BarPlotter(CategoryPlotter):
   
    def plot_markers(self, ax, position, _, row, marker_args, aesthetic_args=None):
        ax.bar(position, row[row['attr']], **marker_args)
        

class PeriStimulusPlotter(FeaturePlotter):
    
    def __init__(self ):
        self.base = self.calc_opts.get('base', 'event') 
        self.pre, self.post = (getattr(self, f"{opt}_{self.base}") for opt in ('pre', 'post'))
        self.marker_names = []
        
    def process_calc(self, info, spec, aesthetics=None, is_last=False, **_):
                
        for row in info:
            attr = spec.get('attr', 'calc')
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


class TextPlotter(PlotterBase):
    """
    Very minimal text plotter.
    """
    def plot(self, cell, text_spec):
        # If cell is an AxWrapper, cell.ax is the actual Matplotlib Ax
        ax = cell.ax if hasattr(cell, 'ax') else cell
        ax.text(0.5, 0.5, text_spec['content'], ha='center', va='center')
        ax.set_axis_off()
        

PLOT_TYPES = {'categorical_scatter': CategoricalScatterPlotter,
              'line_plot': LinePlotter,
              'bar_plot': BarPlotter,
              'waveform': WaveformPlotter,
              'categorical_line': CategoricalLinePlotter,
              'raster': RasterPlotter,
              'psth': PeriStimulusHistogramPlotter}  
