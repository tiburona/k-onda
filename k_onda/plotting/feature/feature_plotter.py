from copy import deepcopy

import matplotlib.pyplot as plt

from ..plotting_helpers import PlottingMixin
from k_onda.base import Base
from k_onda.utils import  recursive_update

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial'] 


class FeaturePlotter(Base, PlottingMixin):

    def process_calc(self, info, spec, ax, aesthetics=None):
        self.apply_borders(ax, spec.get('border', {}))
        if spec.get('aspect'):
            ax.set_box_aspect(spec['aspect'])
        aesthetics = deepcopy(aesthetics) if aesthetics else None
        for row in info:
            aesthetic_args = self.get_aesthetic_args(row, aesthetics)
            val = row[spec.get('attr', 'calc')]
            self.plot_row(ax, val, aesthetic_args)            
    
    def get_aesthetic_args(self, row, aesthetics):

        aesthetic = {}
        aesthetic_spec = deepcopy(aesthetics)

        default, override, invariant = (aesthetic_spec.pop(k, {}) for k in ['default', 'override', 'invariant'])

        aesthetic.update(default)
            
        for category, members in aesthetic_spec.items():
            for member, aesthetic_vals in members.items():
                if self.is_condition_met(category, row, member):
                    recursive_update(aesthetic, aesthetic_vals)

        for combination, overrides in override.items():
            pairs = list(zip(combination.split('.')[::2], combination.split('.')[1::2]))
            if all(row.get(key, val) == val for key, val in pairs):
                aesthetic.update(overrides)

        aesthetic = recursive_update(aesthetic, invariant)

        return aesthetic
    
    def is_condition_met(self, category, row, member):
        if category in row and row[category] == member:
            return True
        for composite_category_type in ['conditions', 'period_types']:
            if {category:member} in row.get(composite_category_type, []):
                return True
        return False
    
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
    