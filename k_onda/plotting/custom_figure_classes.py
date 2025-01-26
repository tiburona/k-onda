import numpy as np
from matplotlib.figure import Figure, SubFigure
from matplotlib.gridspec import GridSpec, SubplotSpec
from matplotlib.transforms import Bbox

def subfigures(obj, nrows=1, ncols=1, squeeze=True, wspace=None, hspace=None,
               width_ratios=None, height_ratios=None,
               left=0, right=1, bottom=0, top=1, **kwargs):
    """
    A replacement for the standard subfigures method that calls obj.add_subfigure,
    which we expect to be overridden in CustomFigure/CustomSubFigure.
    """
    gs = GridSpec(nrows=nrows, ncols=ncols, figure=obj,
                  wspace=wspace, hspace=hspace,
                  width_ratios=width_ratios, height_ratios=height_ratios,
                  left=left, right=right, bottom=bottom, top=top)

    sfarr = np.empty((nrows, ncols), dtype=object)
    for i in range(nrows):
        for j in range(ncols):
            sfarr[i, j] = obj.add_subfigure(gs[i, j], **kwargs)

    # If not using a layout engine, might need to handle spacing manually
    if (obj.get_layout_engine() is None) and (wspace is not None or hspace is not None):
        bottoms, tops, lefts, rights = gs.get_grid_positions(obj)
        for sfrow, bottom_val, top_val in zip(sfarr, bottoms, tops):
            for sf, left_val, right_val in zip(sfrow, lefts, rights):
                bbox = Bbox.from_extents(left_val, bottom_val, right_val, top_val)
                sf._redo_transform_rel_fig(bbox=bbox)

    if squeeze:
        if sfarr.size == 1:
            return sfarr.flat[0]
        sfarr = sfarr.squeeze()

    return sfarr
        

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


class CustomFigure(Figure):
    def add_subfigure(self, spec, **kwargs):
        """
        Instead of creating a standard SubFigure,
        create our custom version.
        """
        return CustomSubFigure(self, spec, **kwargs)

    def subfigures(self, nrows=1, ncols=1, **kwargs):
        print("CustomFigure.subfigures called.")
        return subfigures(self, nrows=nrows, ncols=ncols, **kwargs)
  

class CustomSubFigure(SubFigure):
    def __init__(self, parent, spec: SubplotSpec, **kwargs):
        super().__init__(parent, spec, **kwargs)

    def add_subfigure(self, spec, **kwargs):
        """
        Same trick: create a CustomSubFigure whenever
        sub-subfigures are requested.
        """
        return CustomSubFigure(self, spec, **kwargs)

    def subfigures(self, nrows=1, ncols=1, **kwargs):
        print("CustomSubFigure.subfigures called.")
        return subfigures(self, nrows=nrows, ncols=ncols, **kwargs)


