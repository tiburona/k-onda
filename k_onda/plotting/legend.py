from matplotlib.gridspec import GridSpec

from .plotting_helpers import reshape_subfigures


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
    
    def create_outer_and_subgrid(self):
        outer_grid, main_slice, cax_slice = self.make_outer_grid() 
        self.figure = outer_grid[main_slice]
        self.figure.cax = outer_grid[cax_slice].add_subplot(111)
        self.processor.figure = self.figure
    
    def make_outer_grid(self):
       
        i = ['top', 'bottom', 'left', 'right'].index(self.colorbar_position)
        outer_dim = [1, 1]
        outer_dim[i//2] += 2
        ratio_string = ['height', 'width'][i//2]
        ratios = [1, .025, .075]
        if not i % 2:
            ratios.reverse()
        # self.figure.subplots_adjust(right=.7)
        outer_grid = self.subfigures(*outer_dim, **{f'{ratio_string}_ratios': ratios})
        location_of_main_figure = [0, 0]
        location_of_colorbar_ax = [0, 0]

        location_of_colorbar_ax[i//2] += 1
        if not i%2:
            location_of_main_figure[i//2] += 2

        # outer_grid[*location_of_colorbar_ax].subplots_adjust(right = .8)

        return (outer_grid, tuple(location_of_main_figure), 
                tuple(location_of_colorbar_ax))
        





    

        



