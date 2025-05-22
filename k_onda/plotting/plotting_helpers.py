from copy import deepcopy

import numpy as np
import re

from k_onda.utils import recursive_update


def reshape_subfigures(subfigures, nrows, ncols):
    """
    Ensures that subfigures is returned as a 2D array with the specified dimensions.

    Parameters:
    ----------
    subfigures : object, list, or array
        The output of the subfigures method, which can be a single SubFigure,
        a 1D list, or an already 2D array of SubFigures.
    nrows : int
        The number of rows for the 2D array.
    ncols : int
        The number of columns for the 2D array.

    Returns:
    -------
    np.ndarray
        A 2D array of subfigures with shape (nrows, ncols).
    """
    if isinstance(subfigures, np.ndarray) and subfigures.ndim == 2:
        # If it's already a 2D array, return as-is
        return subfigures
    elif not isinstance(subfigures, (list, np.ndarray)):
        # If it's a single SubFigure, wrap it in a 2D array
        return np.full((nrows, ncols), subfigures)
    else:
        # If it's 1D, reshape to the specified dimensions
        subfigures = np.array(subfigures)
        if subfigures.ndim == 1:
            if subfigures.size != nrows * ncols:
                raise ValueError(f"Cannot reshape {subfigures.size} subfigures into ({nrows}, {ncols})")
            return subfigures.reshape((nrows, ncols))
        else:
            raise ValueError("Input subfigures is not 1D or 2D, which is unexpected.")


def is_condition_met(category, member, entry=None):
        """`self.construct_spec_based_on_conditions` expects this method to be defined"""
        if entry is None:
            return
        if category in entry and entry[category] == member:
            return True
        for composite_category_type in ['conditions', 'period_types']:
            if entry.get(composite_category_type, {}).get(category) == member:
                return True
        return False

class PlottingMixin:

    def construct_spec_based_on_conditions(self, spec_dict, entry=None):
        config_keys = ['default', 'conditional', 'positional', 'override', 'invariant']

        # there's no varying configuration; all the config should be applied to all elements
        if not any(key in spec_dict for key in config_keys):
            return spec_dict
        
        # treat all keys not in one of the config keys as an implicit default (default by default)
        spec_to_return = {k: v for k, v in spec_dict.items() if k not in config_keys}

        default, conditional, positional, override, invariant = (
            spec_dict.get(k, {}) for k in config_keys)

        recursive_update(spec_to_return, default)

        for category, members in conditional.items():
            for member, vals in members.items():
                if is_condition_met(category, member, entry=entry):
                    recursive_update(spec_to_return, vals)

        if entry['cell']:
            for position, members in positional.items():
                if isinstance(position, tuple):
                    if entry['cell'].is_in_extreme_position(
                        position[0], 'last' in position[1], 'absolute' in position[1]):
                        recursive_update(spec_to_return, members)

        for combination, overrides in override.items():
            pairs = list(zip(combination.split('|')[::2], combination.split('|')[1::2]))
            if all(self.is_condition_met(category, member) for category, member in pairs):
                recursive_update(spec_to_return, overrides)

        recursive_update(spec_to_return, invariant)

        return spec_to_return
    
    def get_default_labels(self):
       
        sps = '(Spikes per Second)'

        adjustment = ''

        for comp in ['evoked', 'percent_change']:
            if self.calc_opts.get(comp):
                adjustment = comp
                if adjustment == 'percent_change':
                    adjustment += ' in'
                adjustment = adjustment.replace('_', ' ')
                if comp == 'percent_change':
                    sps = ''
        
        base = self.calc_opts.get('base') if self.calc_opts.get('base') else ''

        label_dict = {'psth': ['Time (s)', 'Normalized Firing Rate'],
              'firing_rates': ['', f'{adjustment} Firing Rate {sps}'],
              'proportion': ['Time (s)', f'Proportion Positive {base.capitalize() + "s"}'],
              'raster': ['Time (s)', f"{self.calc_opts.get('base', 'event').capitalize()}s"],
              'autocorr': ['Lags (s)', 'Autocorrelation'],
              'spectrum': ['Frequencies (Hz)', 'One-Sided Spectrum'],
              'cross_correlations': ['Lags (s)', 'Cross-Correlation'],
              'correlogram':  ['Lags (s)', 'Spikes'],
              'waveform': ['', '']}

        
        for vals in label_dict.values():
            for i, label in enumerate(vals):
                vals[i] = smart_title_case(label) 

        return label_dict
    
      
    @staticmethod
    def divide_data_sources_into_sets(data_sources, max):
        sets = []
        counter = 0
        while counter < max:
            sets.append(data_sources[counter:counter + max])
            counter += max  
        return sets


class LabelMethods:


    
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