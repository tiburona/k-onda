from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator


import numpy as np

from ..plotting_helpers import PlottingMixin
from k_onda.core import Base

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial'] 


class FeaturePlotter(Base, PlottingMixin):

    def set_values(self, calc_config):
        self.info, self.plot_type, self.spec, self.spec_type, self.aesthetics, self.legend_info_list = (
            calc_config.get(k) 
            for k in ('info', 'plot_type', 'spec', 'spec_type', 'aesthetics', 'legend_info_list'))

    def process_calc(self, calc_config):
        self.set_values(calc_config)
        self.aesthetics = deepcopy(self.aesthetics) if self.aesthetics else None

        legend = self.info[0]['last_spec'].get('legend', {})
        cells_with_legend = []

        for i, entry in enumerate(self.info):
            cell = entry['cell']

            aesthetic_args = self.get_aesthetic_args(entry)
            ax_args = aesthetic_args.get('ax', {})
            
            # Plot entry with a temporary label
            val = entry[entry['attr']]
            self.plot_entry(cell, val, aesthetic_args)
            self.apply_ax_args(cell, ax_args, i)

            if legend:
                entry['handle'] = self.get_handle(entry)
                entry['label'] = self.get_entry_label(entry, legend)
                index_key = ','.join(str(d) for d in entry['index'])
                if index_key in legend or 'all' in legend:
                    cells_with_legend.append((cell, index_key))

           
        if legend:
            self.make_legend(set(cells_with_legend), legend)

    def make_legend(self, cells_with_legend, legend):
        handles, labels = [set([e[key] for e in self.info]) 
                           for key in ['handle', 'label']]
        
        for ax, index_key in cells_with_legend:
            legend_key = legend.get(index_key, {})
            legend_key['loc'] = legend_key.get('loc') or 'lower center'
            legend_key['bbox_to_anchor'] = legend_key.get('bbox_to_anchor') or (.8, .9)
            ax.legend(handles, labels, **legend_key)

    def get_handle(self, entry):
        if self.plot_type in ['psth', 'bar_plot']:
            return self.make_bar_handles_from_entry(entry)
        if self.plot_type in ['categorical_line', 'vertical_line']:
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

    def plot_entry(self, ax, val, aesthetic_args=None):
        # Pass a temporary label to ensure the legend can register the line
        ax.plot(np.arange(len(val)), val, label='', **aesthetic_args.get('marker', {}))
        
    def is_condition_met(self, category, member, entry=None):
        """`self.construct_spec_based_on_conditions` expects this method to be defined"""
        if entry is None:
            return
        if category in entry and entry[category] == member:
            return True
        for composite_category_type in ['conditions', 'period_types']:
            if {category:member} in entry.get(composite_category_type, []):
                return True
        return False
        
    def get_entry_label(self, entry, legend):
        relevant_divider_types = legend.get('divisions')
        if relevant_divider_types is None:
            relevant_divider_types = [d['division_type'] for d in entry['last_spec']['divisions']]
        label = []
        for division in relevant_divider_types:
            member = entry[division]
            if division in ['conditions', 'period_types']:
                key, val = list(member.items())[0]  # Correct unpacking
                if entry[key] == val:
                    label.append(val)
            else:
                if entry[division] == member:
                    label.append(member)
        label = ' '.join(label)

        return label
    
    def get_aesthetic_args(self, entry):
        return self.construct_spec_based_on_conditions(self.aesthetics, entry=entry)
    
    def get_tick_args(self, aesthetic_args):
        
        tick_args = {'fontsize': 12}
        tick_args.update(aesthetic_args.get('tick', {}))
        return tick_args
    
    def apply_ax_args(self, cell, ax_args, i):
        ax_list = cell.ax_list if hasattr(cell, 'break_axes') else [cell.obj]
        for ax in ax_list:
            if self.spec_type == 'segment' and i > 0:
                return
            if 'border' in ax_args:
                self.apply_borders(ax, ax_args['border'])
            if 'aspect' in ax_args:
                ax.set_box_aspect(ax_args['aspect'])
            if 'axis_position' in ax_args:
                for side in ax_args['axis_position']:
                    ax.spines[side].set_position(ax_args['axis_position'][side])
            if 'tick_labels' in ax_args:
                self.apply_tick_label_formatting(ax, ax_args['tick_labels'])

    def apply_tick_label_formatting(self, ax, tick_labels):

        for axis in tick_labels:  # axis is 'x' or 'y'
            ax_obj = getattr(ax, f"{axis}axis") 

            opts = tick_labels[axis]

            if opts.get('only_whole_numbers', False):
                ax_obj.set_major_locator(MaxNLocator(integer=True))

            if opts.get('formatter'):
                ax_obj.set_major_formatter(FuncFormatter(eval(opts['formatter'])))

            if opts.get('labelsize'):
                ax.tick_params(axis=axis, labelsize=opts['labelsize'])

            if opts.get('rotation'):
                ax_obj.set_tick_params(rotation=opts['rotation'])

            # TODO add number of decimal formatter
            #ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # 2 decimal places
    
    def apply_borders(self, ax, border):
        border = deepcopy(border)

        default = border.pop('default', {})
        if default:
            for side in ['top', 'bottom', 'left', 'right']:
                self.set_border_args(ax, side, default)

        for side, border_args in border.items():
            self.set_border_args(ax, side, border_args)

    def set_border_args(self, ax, side, border_args):
        # Define falsy values to interpret
        is_falsy = {'f', 'F', False, 'false', 'False', 0}
        
        # Handle visibility settings
        visible = border_args.pop('visible', None)
        if visible:
            # Ensure visible is iterable and convert to a list (e.g., 'fff' → ['f', 'f', 'f'])
            visible = list(visible) if isinstance(visible, (str, tuple, list)) else [visible]
            
            # Pad or truncate visible to ensure it's exactly three items (spine, ticks, labels)
            visible = (visible + [True] * 3)[:3]
            spine_visible, tick_visible, label_visible = visible

            sides = ['top', 'bottom', 'left', 'right'] if side == 'all' else [side]

            for side in sides:

                if sides == ['bottom']:
                    a = 'foo'

                is_truthy = lambda x: x not in is_falsy

                ax.spines[side].set_visible(is_truthy(spine_visible))

                ax.tick_params(**{side: is_truthy(tick_visible)})

                # Determine axis based on the side
                axis = 'x' if side in ['top', 'bottom'] else 'y'
                ax.tick_params(axis=axis, which='both', **{'label' + side: is_truthy(label_visible)})
                
                ax.spines[side].set(**border_args)

    