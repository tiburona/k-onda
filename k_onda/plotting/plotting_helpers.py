import re


def smart_title_case(s):
    lowercase_words = {'a', 'an', 'the', 'at', 'by', 'for', 'in', 'of', 'on', 'to', 'up', 'and', 
                       'as', 'but', 'or', 'nor', 'is'}
    acronyms = {'psth', 'pl', 'hpc', 'bla', 'mrl', 'il', 'bf', 'mua'}
    tokens = re.findall(r'\b\w+\b|[^\w\s]', s)  # Find words and punctuation separately
    title_words = []

    for i, word in enumerate(tokens):
        if word.lower() in lowercase_words and i != 0 and i != len(tokens) - 1:
            title_words.append(word.lower())
        elif word.lower() in acronyms:
            title_words.append(word.upper())
        elif not word.isupper():
            title_words.append(word.capitalize())
        else:
            title_words.append(word)

    # Join words carefully to avoid adding spaces before parentheses
    title = ''
    for i in range(len(title_words)):
        if i > 0 and title_words[i] not in {')', ',', '.', '!', '?', ':'} and title_words[i - 1] not in {'(', '-', '/'}:
            title += ' '
        title += title_words[i]
    return title

class PlottingMixin:
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


def format_label(label, row):
    parts = []  # This will hold the parts of the title

    # Helper function to check if an element is iterable (and not a string)
    def is_iterable(elem):
        try:
            iter(elem)
        except TypeError:
            return False
        return not isinstance(elem, str)  # Exclude strings, as they're iterable in Python
    
    # Iterate through the elements of the title
    for elem in label:
        if elem in row:
            parts.append(row[elem])
        elif is_iterable(elem):
            if is_iterable(elem):  # If the element is a list or tuple, get attributes progressively
                obj = row['data_source']
                for attr in elem[:-1]:
                    obj = getattr(obj, attr)
                parts.append(getattr(obj, elem[-1]))  # Append the final value as a string
        else:
            parts.append(elem)

    # Capitalize each part and join with spaces
    formatted_title = " ".join(smart_title_case(part) for part in parts)
    return formatted_title



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