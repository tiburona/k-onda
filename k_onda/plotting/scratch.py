import matplotlib.pyplot as plt
from matplotlib.figure import SubFigure

class SubFigureWithMargins:
    """
    A hacky wrapper that creates two levels of subfigure:
    - An 'outer' subfigure that sits in the parent at a specified bounding box.
      (This can hold a suptitle, effectively sitting above/below/aside.)
    - An 'inner' subfigure that occupies only part of the outer subfigure
      (so you get a margin around it).

    Usage example:
        sf = SubFigureWithMargins(
            parent_figure_or_subfigure,
            rect=(0.1, 0.1, 0.8, 0.8),  # bounding box in parent's coordinates
            left=0.15, bottom=0.15, right=0.95, top=0.85
        )
        sf.suptitle("I am the subfigure's title!")
        ax = sf.add_subplot()
        ax.plot(...)
    """

    def __init__(
        self, parent, rect=(0.1, 0.1, 0.8, 0.8),
        left=0.1, right=0.9, bottom=0.1, top=0.9,
        outer_subfig_kwargs=None,
        inner_subfig_kwargs=None
    ):
        """
        Parameters
        ----------
        parent : Figure or SubFigure
            The parent in which to nest this subfigure.
        rect : tuple, default (0.1, 0.1, 0.8, 0.8)
            [x, y, width, height] in parent's normalized coordinates.
        left, right, bottom, top : float
            Fractions [0..1] of the outer subfigure's area to allocate
            to the *inner* subfigure. The space outside those edges
            forms the margins.
        outer_subfig_kwargs : dict
            Extra kwargs passed to parent.add_subfigure(...) for the outer subfigure.
        inner_subfig_kwargs : dict
            Extra kwargs passed to outer_subfig.add_subfigure(...) for the inner subfigure.
        """

        if outer_subfig_kwargs is None:
            outer_subfig_kwargs = {}
        if inner_subfig_kwargs is None:
            inner_subfig_kwargs = {}

        # 1) Create a GridSpec in the parent to define the bounding box
        #    for the *outer* subfigure in parent's coordinates.
        self._outer_gs = parent.add_gridspec(
            nrows=1, ncols=1,
            left=rect[0],
            bottom=rect[1],
            right=rect[0] + rect[2],
            top=rect[1] + rect[3]
        )
        # 2) Create the outer subfigure
        self.outer_subfig = parent.add_subfigure(self._outer_gs[0], **outer_subfig_kwargs)

        # 3) Inside that outer subfigure, create another GridSpec that uses [left..right, bottom..top]
        #    but now these coords are relative to the outer subfig's area.
        self._inner_gs = self.outer_subfig.add_gridspec(
            nrows=1, ncols=1,
            left=left, right=right, bottom=bottom, top=top
        )

        # 4) Create the inner subfigure which actually holds the plots
        self.inner_subfig = self.outer_subfig.add_subfigure(self._inner_gs[0], **inner_subfig_kwargs)

    def suptitle(self, *args, **kwargs):
        """
        Place a title on the 'outer' subfigure, so that it sits
        in the margin space above (or below, etc.) the inner subfigure.
        """
        return self.outer_subfig.suptitle(*args, **kwargs)

    def add_subplot(self, *args, **kwargs):
        """
        Add a subplot to the *inner* subfigure. This is where you actually plot.
        """
        return self.inner_subfig.add_subplot(*args, **kwargs)

    def __getattr__(self, name):
        """
        Forward any attributes/methods we haven't explicitly overridden
        to the *inner* subfigure. For example, calling:
            my_nested_subfig.set_facecolor('yellow')
        will apply to the inner subfigure by default.
        """
        return getattr(self.inner_subfig, name)

def demonstration():
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 6))

    fig_gridspec = fig.add_gridspec(1, 1)

    top_subfig = fig.add_subfigure(fig_gridspec[(0, 0)])

    top_subfig.suptitle("I am the top subfig")

    top_subfig_gridspec = top_subfig.add_gridspec(1, 1, top=.7)

    nested_subfig = top_subfig.add_subfigure(top_subfig_gridspec[(0, 0)])

    nested_subfig.suptitle("I am the nested subfig")

plt.show()




# Uncomment to run:
demonstration()
