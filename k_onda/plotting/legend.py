from matplotlib.gridspec import GridSpec


class SubFigureWrapper:
    def __init__(self, parent_fig, gridspec_pos, colorbar_position="right", ratios=None):
        """
        A wrapper for a SubFigure with a dedicated internal SubFigure for the main content.

        Parameters:
        - parent_fig: The parent Figure object.
        - gridspec_pos: Position of this SubFigure in the parent GridSpec (e.g., gs[0, 0]).
        - colorbar_position: One of 'top', 'bottom', 'left', 'right'.
        """
        self.colorbar_position = colorbar_position
        self.parent_fig = parent_fig

        # Create the parent SubFigure
        self.subfig = parent_fig.add_subfigure(gridspec_pos)

        # Create the GridSpec for the internal layout
        if colorbar_position in ["left", "right"]:
            self.gs = GridSpec(1, 2, width_ratios=[0.05, 0.95] if colorbar_position == "left" else [0.95, 0.05])
        elif colorbar_position in ["top", "bottom"]:
            self.gs = GridSpec(2, 1, height_ratios=[0.05, 0.95] if colorbar_position == "top" else [0.95, 0.05])
        else:
            raise ValueError("Invalid colorbar_position. Must be 'top', 'bottom', 'left', or 'right'.")

        # Create the internal SubFigure in the "main" compartment
        # Create the color bar axis in the secondary compartment
        if colorbar_position in ["left", "right"]:
            self.main_subfig = self.subfig.add_subfigure(self.gs[0, 0 if colorbar_position == "right" else 1])
            self.cax = self.subfig.add_subplot(self.gs[0, 1 if colorbar_position == "right" else 0])
        elif colorbar_position in ["top", "bottom"]:
            self.main_subfig = self.subfig.add_subfigure(self.gs[1 if colorbar_position == "top" else 0, 0])
            self.cax = self.subfig.add_subplot(self.gs[0 if colorbar_position == "top" else 1, 0])

    def add_colorbar(self, img, **kwargs):
        """
        Add a color bar to the color bar axis (cax).

        Parameters:
        - img: The image or mappable object to attach the color bar to.
        - kwargs: Additional arguments passed to `self.main_subfig.colorbar`.
        """
        return self.subfig.colorbar(img, cax=self.cax, **kwargs)
    
    def __getattr__(self, name):
        """
        Delegate attribute access to the main SubFigure.
        """
        return getattr(self.main_subfig, name)


class ColorbarMixin:

    @property
    def has_colorbar(self):
        return bool(self.colorbar_spec)

    @property
    def colorbar_spec(self):
        return self.spec and self.spec.get('legend', {}).get('colorbar', {})
           
    @property
    def colorbar_for_each_plot(self):
        return self.has_colorbar and self.colorbar_spec.get('share') in ['each', None]

    @property
    def global_colorbar(self):
        return self.has_colorbar and self.colorbar_spec.get('share') == 'global'

    @property
    def colorbar_position(self):
        return self.colorbar_spec.get('position')

    def colorbar_enabled_subfigure(self, figure, gridspec_slice):
        return SubFigureWrapper(figure, gridspec_slice, colorbar_position=self.colorbar_position)
    
    def make_outer_gridspec(self):
       
    
        i = ['top', 'bottom', 'left', 'right'].index(self.colorbar_position)
        outer_gridspec_dim = [1, 1]
        outer_gridspec_dim[i//2] += 1
        ratio_string = ['height', 'width'][i//2]
        ratios = [.9, .1]
        if not i % 2:
            ratios.reverse()
        outer_gs = self.figure.add_gridspec(*outer_gridspec_dim, **{f'{ratio_string}_ratios': ratios})
        location_of_main_figure_gridspec = [0, 0]
        location_of_colorbar_ax = [0, 0]
        if i % 2:
            location_of_colorbar_ax[i//2] += 1
        else:
            location_of_main_figure_gridspec[i//2] +=1

        return (outer_gs, tuple(location_of_main_figure_gridspec), 
                tuple(location_of_colorbar_ax))
        





    

        



