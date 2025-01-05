from matplotlib.font_manager import FontProperties 

from .plotting_helpers import format_label


class LabelMethods:

    def label(self, row, ax, aesthetics, is_last):
        # label is a dictionary like {'component': {'axis_label': (), 'title': ''}, 'ax': {}}
        layout = self.active_layout
        label_properties = aesthetics.get('label', {})
        font_properties = FontProperties(aesthetics.get('font_properties', {}))

        for position in label_properties:
            if position == 'component' and is_last:
                axis_labels, title = self.get_labels(label_properties['component'], row)
                xy = label_properties['component'].get('xy', [(0.5, 0), (0.025, 0.5)])
                for text, coords, axis, dim in zip(axis_labels, xy, ['ha', 'va'], ['x', 'y']):
                    rotation = 90 if dim == 'y' else 0  
                    kwargs = {
                        'xy': coords,
                        'xycoords': 'axes fraction',
                        axis: 'center',
                        'font_properties': font_properties,
                        'rotation': rotation
                    }
                    # Apply rotation after creation
           
                    label = layout.frame_ax.annotate(text, **kwargs)
                    self.adjust_label_position(layout.frame_ax, label, axis=dim)
                if title:
                    layout.frame_ax.set_title(title)

            elif position == 'ax':
                axis_labels, title = self.get_labels(label_properties['ax'], row)
                ax.set_title(title)
                ax.set_xlabel(axis_labels[0])
                ax.set_ylabel(axis_labels[1])

    def get_labels(self, lab_info, row):
        lab_info.update(self.active_spec['aesthetics'].get('label', {}))
        axis = lab_info.get('axis', '')
        if axis: 
            if axis == 'default':
                axis_labels = self.get_default_labels()[self.calc_type]
            else:
                axis_labels = [lab if isinstance(lab, str) else lab for lab in axis]
        else:
            axis_labels = ('', '')
           
        title = lab_info.get('title', '')
        if title:
            title = format_label(lab_info['title'], row)
        return axis_labels, title
    
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