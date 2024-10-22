from multiprocessing.dummy.connection import families
import os
import math
import numpy as np
import seaborn as sns
import pandas as pd
from copy import deepcopy, copy
import json
from collections import defaultdict, deque

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch
import matplotlib.ticker as ticker



from math_functions import get_positive_frequencies, get_spectrum_fenceposts, nearest_power_of_10
from plotting_helpers import smart_title_case, PlottingMixin
from utils import to_serializable
from base_data import Base
from stats import Stats
from phy_interface import PhyInterface
from plotter_base import PlotterBase
from partition import Section, Segment, Subset
from subplotter import Figurer




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
            self.close_plot()

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
        fig.savefig(os.path.join(path, fname), bbox_inches='tight')
        opts_filename = fname.replace('png', 'txt')

        with open(os.path.join(path, opts_filename), 'w') as file:
            json.dump(to_serializable(self.calc_opts), file)
        plt.close(fig)

    def delegate(self, info, is_last=False):

        def send(plot_type):
            PLOT_TYPES[plot_type]().process_calc(info, main=main, aesthetics=aesthetics, 
                                                 is_last=is_last)

        aesthetics = self.active_spec.get('aesthetics', {})
        main = True
        if 'layers' in self.active_spec:
            for layer in self.active_spec['layers']:                
                main = layer.get('main', True)
                aesthetics.update(layer.get('aesthetics', {}))
                if 'attr' in layer:
                    self.active_spec['attr'] = layer['attr']
                if 'plot_type' in layer:
                    send(layer['plot_type'])
                else:
                    send(self.graph_opts['plot_type'])
        else:
            send(self.graph_opts['plot_type'])
                

class FeaturePlotter(PlotterBase, PlottingMixin):
    
    def get_aesthetics_args(self, row, aesthetics):

        aesthetic = {}
        aesthetic_spec = deepcopy(aesthetics)
        default, override = (aesthetic_spec.pop(k, {}) for k in ['default', 'override'])

        aesthetic.update(default)
            
        for category, members in aesthetic_spec.items():
            for member, aesthetic_vals in members.items():
                if category in row and row[category] == member:
                    aesthetic.update(aesthetic_vals)

        for combination, overrides in override.items():
            pairs = zip(combination.split('_')[::2], combination.split('_')[1::2])
            if all(row.get(key, val) == val for key, val in pairs):
                aesthetic.update(overrides)

        return aesthetic
    
    def handle_broken_axes(self, data):
    # Initial data division (copying the original data)
        data_divisions = np.array([copy(data)])
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
                        # Check if dd is 1D and reshape it if needed
                        dd[:, self.get_data_slice(arg_set)] 
                        if dd.ndim > 1 else dd.reshape(1, -1)[:, self.get_data_slice(arg_set)]
                        for dd in data_divisions
                        for arg_set in acks.break_axes[0]
                ]
            x_slices = acks.break_axes[0]
        else:
            x_slices = [0, int(len(data[0])/self.bin_size)]
            ax_list = [self.active_acks]
        
        return data_divisions, ax_list, x_slices
    
    def get_data_slice(self, arg_set):
        """Return a slice object for the given arg_set and bin_size."""
        return slice(*((arg_set) / self.bin_size).astype(int))
    

class HistogramPlotter(FeaturePlotter):
    
    def plot_hist(self, x, y, width, acks):
        acks.ax.bar(x, y, width=width) 


class CategoryPlotter(FeaturePlotter):

    # @property
    # def cat_width(self):
    #     return self.active_spec.get('cat_width', .8)

    def find_position(self, observation, aesthetics, division_types=None, start_position=0):
        segment_info = self.active_spec['divisions']
        if division_types is None:
            division_types = deque(sorted(
                [k for k in segment_info.keys() if 'grouping' in segment_info[k]],
                key=lambda x: segment_info[x]['grouping']))

        # some_factor adjusts how much space each character takes
        labels = segment_info[division_types[0]]['members']
        spacing = self.get_aesthetics_args(observation, aesthetics).get('spacing', 1)
        label_lengths = [len(label) for label in labels]
        max_label_length = max(label_lengths)
        spacing = max_label_length * spacing

        mult_factor = 1
        for dt in division_types:
            num_members = len(segment_info[dt]['members'])
            mult_factor *= num_members + 2 * spacing

        current_division_type = division_types.popleft()
        value = observation[current_division_type]
        index = segment_info[current_division_type]['members'].index(value)
        position = index * mult_factor + start_position

        if len(division_types) == 0:
            return position + spacing
        else:
            return self.find_position(
                observation, division_types=division_types, start_position=position)

    def compute_outer_label_positions(inner_positions, outer_labels, num_inner_per_outer):
        """
        Compute positions and labels for the outer grouping level.

        Parameters:
        - inner_positions (list of float): Positions of the inner level ticks.
        - outer_labels (list of str): Labels for the outer grouping level.
        - num_inner_per_outer (int or list of int): Number of inner positions per outer group.
            If an integer, it's assumed the same for all outer groups.
            If a list, it should have the same length as outer_labels.

        Returns:
        - outer_positions (list of float): Computed positions for outer group labels.
        - outer_labels (list of str): Labels for the outer grouping level.
        """
        outer_positions = []
        idx = 0  # Index to track position in inner_positions

        if isinstance(num_inner_per_outer, int):
            num_inner_per_outer = [num_inner_per_outer] * len(outer_labels)
        elif len(num_inner_per_outer) != len(outer_labels):
            raise ValueError("Length of num_inner_per_outer must match length of outer_labels.")

        for count in num_inner_per_outer:
            # Get the positions corresponding to the current outer group
            group_positions = inner_positions[idx:idx + count]
            # Compute the midpoint of these positions
            group_midpoint = sum(group_positions) / len(group_positions)
            outer_positions.append(group_midpoint)
            idx += count  # Move to the next set of positions

        return outer_positions
    
    def label(self, positions):
        ax = self.active_acks
        for i, (dim, edge) in enumerate(zip(['x', 'y'], ['bottom', 'left'])):
            if ax.index[i] == 0 or getattr(ax, f"{edge}_edge"):
                # TODO: want this to be responsive to label_ax/label_component
                getattr(ax, f"set_{dim}label")(
                    self.get_labels()[self.calc_type][i])   
        
        tick_label_bbox = ax.get_xticklabels()[0].get_window_extent()
        bbox_in_ax = tick_label_bbox.transformed(ax.transAxes.inverted())
        tick_label_ymin = bbox_in_ax.ymin

        divisions = self.active_spec['divisions']
        levels = reversed(sorted(divisions.keys(), key=lambda k: divisions[k]['grouping']))
        level_adjustment = 0
        for i, level in enumerate(levels):
            labels = divisions[level]['members']
            if 'legend' not in divisions[level]:
                level_adjustment -= .05
            if i == 0:
                ax.set_xticks(positions)
                ax.set_xticklabels(labels)
                inner_positions = ax.get_xticks
            else:
                num_inner_per_outer = int(len(labels)/self.active_spec[levels[i-1]]['members'])
                current_positions = self.compute_outer_label_positions(
                    inner_positions, labels, num_inner_per_outer)
                for lab, pos in zip(labels, current_positions):
                    ax.text(pos, tick_label_ymin - level_adjustment, lab)
                inner_positions = current_positions


class CategoricalScatterPlotter(CategoryPlotter):

    def process_calc(self, info, main=True, aesthetics=None, is_last=False):
        self.cat_width = aesthetics.get('default', {}).get('cat_width', .8)
        ax = self.active_acks
        positions = []
        for row in info:
            if self.active_spec_type == 'segment':
                position = self.find_position(row, aesthetics) + self.cat_width/2
                positions.append(position)
            else:
                # do something else
                pass
            scatter_vals = row[self.active_spec['attr']] 
            jitter = np.random.rand(len(scatter_vals)) * self.cat_width/2 - self.cat_width/4
            aesthetic_args = self.get_aesthetics_args(row, aesthetics)
            marker_args = {k: v for k, v in aesthetic_args.items() if k in ['color']}
            if 'background_color' in aesthetic_args:
                background_color, alpha = aesthetic_args.pop('background_color')
                ax.axvspan(
                    position - self.cat_width/2, position + self.cat_width/2, 
                    facecolor=background_color, alpha=alpha)
            ax.scatter([position + j for j in jitter], scatter_vals, **marker_args)
    
        if main:
            self.label(positions)

       
class LinePlotter(FeaturePlotter):
    def process_calc(self, info, aesthetics=None, **_):
        attr = self.active_spec['attr']
        ax = self.active_acks
        for row in info:
            val = row[attr]
            ax.plot(np.arange(len(val)), val, **self.get_aesthetics_args(row, aesthetics))


class WaveformPlotter(LinePlotter):
    pass


class CategoricalLinePlotter(CategoryPlotter):
    def process_calc(self, info, aesthetics=None, **_):
        self.cat_width = aesthetics.get('default', {}).get('cat_width', .8)
        ax = self.active_acks
        names = ['linestyles', 'colors']

        for row in info:
            position = self.find_position(row, aesthetics) + self.cat_width/2
            aesthetic_args = self.get_aesthetics_args(row, aesthetics)
            divisor = aesthetic_args.pop('divisor', 2)
            marker_args = {name: aesthetic_args[name] for name in names if name in aesthetic_args}
            ax.hlines(row['mean'], position-self.cat_width/divisor, position+self.cat_width/divisor, 
                      **marker_args)

class BarPlotter(CategoryPlotter):
      def process_calc(self, info):

        self.cat_width = self.graph_opts.get('cat_width', .2)
        
        for row in info:
            aesthetic_args = self.get_aesthetics_args(row)
            position = self.find_position(row, self.active_spec)
            bar = self.active_acks.bar(position, getattr(row['data_source'], row['attr']), 
                                     self.cat_width, **aesthetic_args)
            

class PeriStimulusPlotter(FeaturePlotter):
    
    def __init__(self ):
        self.base = self.calc_opts.get('base', 'event') 
        self.pre, self.post = (getattr(self, f"{opt}_{self.base}") for opt in ('pre', 'post'))
        self.marker_names = []
        
    def process_calc(self, info, aesthetics=None, is_last=False, **_):
        
        acks = self.active_acks
        
        for row in info:
            data = row[self.active_spec['attr']]
            aesthetic_args = self.get_aesthetics_args(row, aesthetics)
            marker_args = {name: aesthetic_args[name] 
                           for name in self.marker_names if name in aesthetic_args}
            data_divisions, ax_list, x_slices = self.handle_broken_axes(data)
            num_rows = len(data)
            
            for ax, data, x_slice in zip(ax_list, data_divisions, x_slices):
                self.plot_row(ax, data, aesthetic_args, marker_args)
                self.set_x_ticks(ax, data, x_slice)
                self.place_marker(ax, aesthetic_args)
                if is_last:
                    self.label()   
                
    def set_x_ticks(self, ax, data, x_slice):
        
        manual_ticks = np.arange(0, len(data[0]) + 1, step=10)  # Adjust step size as needed
        ax.set_xticks(manual_ticks)

        # Get the existing tick positions (in bins)
        existing_ticks = ax.get_xticks()

        # Filter the ticks to only those within the visible range of the data (i.e., corresponding to x_slice)
        visible_ticks = [tick for tick in existing_ticks if 0 <= tick <= len(data[0])]

        # Calculate the start and end time for the current x_slice (in seconds)
        x_slice_start_time = x_slice[0] - self.pre  
        x_slice_end_time = x_slice[1] - self.pre

        # Create a time range that matches the visible ticks, from x_slice_start_time to x_slice_end_time
        tick_range = np.linspace(x_slice_start_time, x_slice_end_time, len(visible_ticks))

        # Set the x-tick positions and labels
        ax.set_xticks(visible_ticks)  # Use only the visible ticks
        ax.set_xticklabels([f"{label:.1f}" for label in tick_range])  # Labels in seconds
        
    def place_marker(self, ax, aesthetic_args):
        marker = aesthetic_args.get('marker', {})
        marker_type = marker.get('type')
        when = marker.get('when', ('pre', 'post'))
        ylim = ax.get_ylim()  # Retrieve ylim only once

        
        if marker_type == 'vertical_line':
            for event in when:
                ax.vlines(getattr(self, f"{event}_{base}")/self.bin_size, ylim[0], ylim[1], 
                            colors='black')
                
         # # Add the gray rectangle patch
                # patch_start = row['data_source'].children[0].zero_point
                # patch_end = patch_start + 0.05
                # ylim = ax.get_ylim()  # Retrieve ylim only once
                # ax.add_patch(plt.Rectangle(
                #     (patch_start, ylim[0]), patch_end, ylim[1] - ylim[0], 
                #     facecolor='gray', alpha=0.3
                # ))
                
    def label(self):
        subplotter = self.active_plotter
        axes = subplotter.get_ax_wrappers()  # Get the actual AxWrapper objects
        aesthetic_args = self.active_spec.get('aesthetic_args', {})
        label_position = aesthetic_args.get('label_position', 'label_component')

        if label_position == 'label_component':
            # Annotate the frame_ax to position the labels outside the ticks
            xlabel = subplotter.frame_ax.annotate(
                self.get_labels()[self.calc_type][0],
                xy=(0.5, 0),  # Center the x-label, push below the ticks
                xycoords='axes fraction',
                ha='center'
            )

            ylabel = subplotter.frame_ax.annotate(
                self.get_labels()[self.calc_type][1],
                xy=(0.025, 0.5),  # Push y-label outside the ticks
                xycoords='axes fraction',
                rotation=90,
                va='center'
            )

            # Adjust the label positions based on the label size
            self.adjust_label_position(subplotter.frame_ax, xlabel, axis='x')
            self.adjust_label_position(subplotter.frame_ax, ylabel, axis='y')
            
    
    def adjust_label_position(self, ax, label, axis='x'):
        # Get the bounding box of the label
        renderer = ax.figure.canvas.get_renderer()
        bbox = label.get_window_extent(renderer=renderer)

        # Get the size of the figure in pixels
        fig_width, fig_height = ax.figure.get_size_inches() * ax.figure.dpi

        if axis == 'x':
            label_width = bbox.width / fig_width  # Normalize width in figure units
            new_x = 0.5 #- label_width/2  # Adjust to center
            label.set_position((new_x, label.get_position()[1]))  # Adjust the x-position
        elif axis == 'y':
            label_height = bbox.height / fig_height  # Normalize height in figure units
            new_y = 0.5 #- label_height/2  # Adjust to center
            label.set_position((label.get_position()[0], new_y))  # Adjust the y-position
                

class RasterPlotter(PeriStimulusPlotter):
    
    def process_calc(self, *args, **kwargs):
        self.marker_names = ['linestyles', 'colors']
        super().process_calc(*args, **kwargs)
                
    def plot_row(self, ax, data, aesthetic_args, marker_args):
        ax.set_ylim(0, len(data))  # Set ylim based on the number of rows
                
        # Plot spikes on the vlines
        for i, spiketrain in enumerate(data):
            for j, spike in enumerate(spiketrain):
                if spike:
                    ax.vlines(j, i, i + line_length, **marker_args)
    
                
class PeriStimulusHistogramPlotter(PeriStimulusPlotter, HistogramPlotter):
        
    def plot_row(self, ax, data, aesthetic_args, marker_args):
        data = data[0]
        x = list(range(0, len(data)))
        y = data
        self.plot_hist(x, y, self.calc_opts['bin_size'], ax)
        


PLOT_TYPES = {'categorical_scatter': CategoricalScatterPlotter,
              'line_plot': LinePlotter,
              'waveform': WaveformPlotter,
              'categorical_line': CategoricalLinePlotter,
              'raster': RasterPlotter,
              'psth': PeriStimulusHistogramPlotter}  


   



