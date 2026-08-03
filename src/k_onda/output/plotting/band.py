from dataclasses import dataclass, replace, field
import matplotlib.patches as patches


from .core import PlotDirective, replace_plot_node


class BandMixin:

    def band(self, x=None, y=None, where="all", color=None, opacity=None, **kwargs):
        band = Band(x=x, y=y, where=where, color=color, opacity=opacity, kwargs=kwargs)
        return AddBand(band)(self)
        

@dataclass(frozen=True)
class Band:
    x: tuple | None = None
    y: tuple | None = None
    where: list | tuple | dict | str = "all"
    color: str | None = None
    opacity: float | None = None
    kwargs: dict = field(default_factory=dict)


class AddBand(PlotDirective):

    def __init__(self, band):
        self.band = band

    def direct(self, input):
        return replace_plot_node(input, overlays=(*input.overlays, self.band))

    
class BandRenderer:

    def build_band_plan(self, plot_node, layout):
        plan = {
            (panel.row, panel.col): []
            for panel in layout.flat_panels
        }

        for overlay in plot_node.overlays:
            if not isinstance(overlay, Band):
                continue
            band = self.resolve_config(overlay)

            for panel in self.select_panels_for_band(layout, band.where):
                plan[(panel.row, panel.col)].append(band)

        return plan

    def select_panels_for_band(self, layout, where):
        flat_panels = layout.flat_panels
        
        if where == "all":
            return flat_panels
        if (
            isinstance(where, (tuple, list)) 
            and len(where) == 2
            and all(isinstance(val, int) for val in where)
            ):
            cell = tuple(where)
            return [panel for panel in flat_panels if cell == (panel.row, panel.col)]
        if isinstance(where, (tuple, list)):
            cells = [tuple(cell) for cell in where]
            return [panel for panel in flat_panels if (panel.row, panel.col) in cells]
        if isinstance(where, dict):
            return [
                panel for panel in flat_panels 
                if all(key in panel.coords and where[key] == panel.coords[key] 
                       for key in where)
                ]
        raise ValueError(f"Unknown value for where {where}")

    def resolve_config(self, band_spec):

        new_kwargs = {**band_spec.kwargs}

        if band_spec.opacity is not None:
            new_kwargs["alpha"] = band_spec.opacity
        if band_spec.color is not None:
            new_kwargs["facecolor"] = band_spec.color

        return replace(band_spec, kwargs=new_kwargs)

    def make_band(self, band_spec, ax):
        x = band_spec.x
        y = band_spec.y

        if x is not None and y is None:
            return ax.axvspan(
                x[0],
                x[1],
                ymin=0,
                ymax=1,
                **band_spec.kwargs,
            )

        if x is None and y is not None:
            return ax.axhspan(
                y[0],
                y[1],
                xmin=0,
                xmax=1,
                **band_spec.kwargs,
            )

        if x is not None and y is not None:
            rect = patches.Rectangle(
                (x[0], y[0]),
                x[1] - x[0],
                y[1] - y[0],
                transform=ax.transData,
                **band_spec.kwargs,
            )
            ax.add_patch(rect)
            return rect

        raise ValueError("A band requires an x or y interval.")