import re

class PlottingMixin:
    def get_labels(self):
       
        sps = '(Spikes per Second)'

        adjustment = ''

        for comp in ['evoked', 'percent_change']:
            if self.calc_spec.get(comp):
                adjustment = comp
                adjustment = smart_title_case(adjustment.replace('_', ' '))
                if comp == 'percent_change':
                    sps = ''
        
        base = self.calc_spec.get('base') if self.calc_spec.get('base') else ''

        return {'psth': ('Time (s)', 'Normalized Firing Rate'),
                'firing_rates': ('', f'{adjustment} Firing Rate {sps}'),
                'proportion': ('Time (s)', ''f'Proportion Positive {base.capitalize() + "s"}'),
                'raster': ('Time (s)', f"{self.calc_spec.get('base', 'event').capitalize()}s"),
                'autocorr': ('Lags (s)', 'Autocorrelation'),
                'spectrum': ('Frequencies (Hz)', 'One-Sided Spectrum'),
                'cross_correlations': ('Lags (s)', 'Cross-Correlation'),
                'correlogram':  ('Lags (s)', 'Spikes')}
    
      
    @staticmethod
    def divide_data_sources_into_sets(data_sources, max):
        sets = []
        counter = 0
        while counter < max:
            sets.append(data_sources[counter:counter + max])
            counter += max  
        return sets
    
    @staticmethod
    def initialize_mins_and_maxes(data_sources):
        mins_and_maxes = {data_source.identifier: {'min': float('inf'), 'max': float('-inf')} 
                          for data_source in data_sources}
        mins_and_maxes['global'] = {'min': float('inf'), 'max': float('-inf')}
        return mins_and_maxes
    
    @staticmethod
    def evaluate_mins_and_maxes(data_source, data, mins_and_maxes):
        data_min, data_max = data.min(), data.max()
        mins_and_maxes[data_source.identifier]['min'] = min(
            mins_and_maxes[data_source.identifier]['min'], data_min)
        mins_and_maxes[data_source.identifier]['max'] = max(
                mins_and_maxes[data_source.identifier]['max'], data_max)
        mins_and_maxes['global']['min'] = min(mins_and_maxes['global']['min'], data_min)
        mins_and_maxes['global']['max'] = min(mins_and_maxes['global']['max'], data_max)

    def set_labels(self, x_and_y_labels=(None, None)):
        canonical_labels = self.get_labels().get(self.calc_type)
        labels = [canonical_labels[i] if label is None else label for i, label in enumerate(x_and_y_labels)]

        if self.plot_type == 'standalone':
            # Create an invisible subplot that spans the entire figure
            big_subplot = self.fig.add_subplot(111, frame_on=False)
            big_subplot.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            object_to_label = big_subplot
            x_coord_of_y_label = -0.08
            object_to_label.yaxis.set_label_coords(x_coord_of_y_label, 0.5)  # For the y-axis

        elif self.plot_type == 'subplot':
            object_to_label = self.ax
        else:
            object_to_label = self.invisible_ax

            if self.__class__.__name__ == 'PeriStimulusPlotter':  # kludge to deal with misplacement of y-axis label
                x_coord_of_y_label = -0.08
            else:
                x_coord_of_y_label = -0.15

            object_to_label.xaxis.set_label_coords(0.5, -0.15)  # For the x-axis
            object_to_label.yaxis.set_label_coords(x_coord_of_y_label, 0.5)  # For the y-axis

        [getattr(object_to_label, f"set_{dim}label")(labels[i], fontsize=15 * self.multiplier)
         for i, dim in enumerate(['x', 'y'])]
        

    def get_color(self, group=None, period_group=None, period_type=None):
        default_color = self.graph_opts.get('default_color', 'black')

        # Construct a key based on the parameters provided
        key_parts = []
        if group not in [None, 'all', '']:
            key_parts.append(f"group {group}")
        if period_group not in [None]:
            key_parts.append(f"period_group {period_group}")
        if period_type not in [None, 'all', '']:
            key_parts.append(f"period_type {period_type}")  # Assume period_type is always specified

        current_context = ' '.join(key_parts).strip()

        # Search for the most specific matching pattern in the selections
        best_match = default_color
        max_match_length = 0
        for pattern, color in self.graph_opts.get('colors', {}).items():
            pattern_parts = pattern.split()
            if all(part in current_context.split() for part in pattern_parts):
                # Prefer the longest, most specific match found
                if len(pattern_parts) > max_match_length:
                    max_match_length = len(pattern_parts)
                    best_match = color

        return best_match


def smart_title_case(s):
    lowercase_words = {'a', 'an', 'the', 'at', 'by', 'for', 'in', 'of', 'on', 'to', 'up', 'and', 'as', 'but', 'or',
                       'nor', 'is'}
    acronyms = {'psth', 'pl', 'hpc', 'bla', 'mrl', 'il', 'bf'}
    words = [w for w in re.split(r'([_\W]+)', s) if w not in  ['_', ' ']]
    title_words = []
    for i, word in enumerate(words):
        if word.lower() in lowercase_words and i != 0 and i != len(words) - 1:
            title_words.append(word.lower())
        elif word.lower() in acronyms:
            title_words.append(word.upper())
        elif not word.isupper():
            title_words.append(word.capitalize())
        else:
            title_words.append(word)
    title = ' '.join(title_words)
    return title


def ac_str(s):
    for (old, new) in [('pd', 'Pandas'), ('np', 'NumPy'), ('ml', 'Matlab')]:
        s = s.replace(old, new)




