from copy import deepcopy
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from ..plotting_helpers import PlottingMixin
from k_onda.base import Base

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial'] 


class FeaturePlotter(Base, PlottingMixin):

    def process_calc(self, calc_config):
        info, spec_type, aesthetics = (
            calc_config[k] for k in ('info', 'spec_type', 'aesthetics'))
        aesthetics = deepcopy(aesthetics) if aesthetics else None
        unique_axs = defaultdict(lambda: {'ax': None, 'entries': []})

        for i, entry in enumerate(info):
            ax = entry['cell']
            unique_axs[id(ax)]['ax'] = ax
            unique_axs[id(ax)]['entries'].append(entry)

            aesthetic_args = self.get_aesthetic_args(entry, aesthetics)
            ax_args = aesthetic_args.get('ax', {})
            self.apply_ax_args(ax, ax_args, i, spec_type)

            # Plot entry with a temporary label
            val = entry[entry['attr']]
            self.plot_entry(ax, val, aesthetic_args)
            entry['legend_label'] = self.get_entry_label(entry)

        self.make_legend(unique_axs)

    def make_legend(self, unique_axs):
        for i, data in enumerate(unique_axs.values()):
            ax = data['ax']
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
                ax.legend(handles, custom_labels, **legend['key'])

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
    
    def get_aesthetic_args(self, entry, aesthetics):
        return self.construct_spec_based_on_conditions(aesthetics, entry=entry)
    
    def apply_ax_args(self, ax, ax_args, i, spec_name):
        if spec_name == 'segment' and i > 0:
            return
        if 'border' in ax_args:
            self.apply_borders(ax, ax_args['border'])
        if 'aspect' in ax_args:
            ax.set_box_aspect(ax_args['aspect'])
        if 'axis_position' in ax_args:
            for side in ax_args['axis_position']:
                ax.spines[side].set_position(ax_args['axis_position'][side])
    
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

            # Set spine visibility
            if spine_visible in is_falsy:
                ax.spines[side].set_visible(False)
            else:
                ax.spines[side].set_visible(True)

            # Set tick visibility
            if tick_visible in is_falsy:
                ax.tick_params(**{side: False})  # Disable bottom ticks

            # Determine axis based on the side
            axis = 'x' if side in ['top', 'bottom'] else 'y'

            # Set label visibility
            if label_visible in is_falsy:
                if label_visible in is_falsy:
                    ax.tick_params(axis=axis, which='both', **{'label' + side: False})
        
        # Apply remaining border arguments to the spine
        ax.spines[side].set(**border_args)
    