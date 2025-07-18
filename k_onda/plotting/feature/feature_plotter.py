from copy import deepcopy
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

import numpy as np

from ..plotting_helpers import PlottingMixin
from k_onda.core import Base

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial'] 


class FeaturePlotter(Base, PlottingMixin):

    def set_values(self, calc_config):
        self.info, self.spec, self.spec_type, self.aesthetics, self.legend_info_list = (
            calc_config.get(k) 
            for k in ('info', 'spec', 'spec_type', 'aesthetics', 'legend_info_list'))

    def process_calc(self, calc_config):
        self.set_values(calc_config)
        self.aesthetics = deepcopy(self.aesthetics) if self.aesthetics else None
        unique_axs = defaultdict(lambda: {'cell': None, 'entries': []})

        for i, entry in enumerate(self.info):
            cell = entry['cell']
            unique_axs[id(cell)]['cell'] = cell
            unique_axs[id(cell)]['entries'].append(entry)

            aesthetic_args = self.get_aesthetic_args(entry)
            ax_args = aesthetic_args.get('ax', {})
            
            # Plot entry with a temporary label
            val = entry[entry['attr']]
            self.plot_entry(cell, val, aesthetic_args)
            self.apply_ax_args(cell, ax_args, i)
            entry['legend_label'] = self.get_entry_label(entry)

        self.make_legend(unique_axs)

    def make_legend(self, unique_axs):
        for i, data in enumerate(unique_axs.values()):
            ax = data['cell']
            entries = data['entries']

            if not entries or 'last_spec' not in entries[0]:
                continue

            legend = entries[0]['last_spec'].get('legend', {})
            if legend and legend.get('which', 'all') in ['all', i]:
                handles = self.get_handles(ax)  # Instead of get_legend_handles_labels()
                if not handles:
                    # Debugging output if handles are empty
                    print("No handles found on axis:", ax)
                    print("Lines on axis:", ax.get_lines())
                print("Manually collected handles:", handles)
                custom_labels = [e['legend_label'] for e in entries]
                print(custom_labels)
                anchor_y = legend.get('anchor_y', 0.9)
                anchor_x = legend.get('anchor_x', 0.8)
                # Combine with any pre-existing legend key parameters
                legend_key = legend.get('key', {}).copy()  # copy to avoid modifying the original
                # Set location and bbox_to_anchor to position the legend above your data.
                legend_key.update({
                    'loc': legend.get('loc', 'lower center'),  
                    'bbox_to_anchor': (anchor_x, anchor_y)
                })
                ax.legend(handles, custom_labels, **legend_key)
              

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
        
    def get_entry_label(self, entry):
        relevant_divisions = entry['last_spec']['divisions']
        label = []
        for division in relevant_divisions:
            for member in division['members']:
                if division['divider_type'] in ['conditions', 'period_types']:
                    key, val = list(member.items())[0]  # Correct unpacking
                    if entry[key] == val:
                        label.append(val)
                else:
                    if entry[division['divider_type']] == member:
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

    