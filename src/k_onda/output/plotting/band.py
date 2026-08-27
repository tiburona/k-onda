from dataclasses import dataclass, replace, field
from numbers import Integral, Real

import matplotlib.patches as patches

from k_onda.utils import validate_types, ValidationMixin

from .core import PlotDirective, replace_plot_node


BandCell = tuple[Integral, Integral] | list[Integral]
BandWhere = str | BandCell | tuple[BandCell, ...] | list[BandCell] | dict[str, object]


class BandMixin:

    @validate_types
    def band(
        self,
        *,
        x: tuple[object, object] | None = None,
        y: tuple[object, object] | None = None,
        where: BandWhere = "all",
        color: str | None = None,
        opacity: Real | None = None,
        **kwargs: object,
    ):
        band = Band(x=x, y=y, where=where, color=color, opacity=opacity, kwargs=kwargs)
        return AddBand(band)(self)
        

@dataclass(frozen=True, kw_only=True)
class Band(ValidationMixin):
    x: tuple[object, object] | None = None
    y: tuple[object, object] | None = None
    where: BandWhere = "all"
    color: str | None = None
    opacity: Real | None = None
    kwargs: dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        self.validate_type_hints()
        self.validate_string_iterable("keyword name", self.kwargs)
        self._validate_intervals()
        self._validate_where()
        self._validate_appearance()

    def _validate_intervals(self):
        if self.x is None and self.y is None:
            raise ValueError(
                f"{self.format_call()}: at least one of x and y is required."
            )

    def _validate_where(self):
        if isinstance(self.where, str):
            if self.where != "all":
                raise ValueError(
                    f"{self.format_call()}: string where must be 'all'."
                )
            return

        if isinstance(self.where, dict):
            self.validate_parameter("where", self.where, nonempty=True)
            self.validate_string_iterable("where coordinate name", self.where)
            return

        self.validate_parameter("where", self.where, nonempty=True)
        # Normalize one cell and a collection of cells to the same shape.
        cells = (self.where,) if isinstance(self.where[0], Integral) else self.where

        for cell in cells:
            for value in cell:
                self.validate_number(
                    "panel cell coordinate",
                    value,
                    number_type=Integral,
                    minimum=0,
                )

    def _validate_appearance(self):
        self.validate_number(
            "opacity",
            self.opacity,
            allow_none=True,
            minimum=0,
            maximum=1,
        )

        reserved_kwargs = set()
        if self.color is not None:
            reserved_kwargs.update(("color", "facecolor"))
        if self.opacity is not None:
            reserved_kwargs.add("alpha")
        if self.x is not None and self.y is None:
            reserved_kwargs.update(("ymin", "ymax"))
        elif self.x is None and self.y is not None:
            reserved_kwargs.update(("xmin", "xmax"))
        else:
            reserved_kwargs.update(("xy", "width", "height", "transform"))

        conflicts = reserved_kwargs & set(self.kwargs)
        if conflicts:
            raise ValueError(
                f"{self.format_call()}: renderer kwargs cannot override values "
                f"managed by the band specification: {sorted(conflicts)!r}."
            )


class AddBand(PlotDirective):

    def __init__(self, band: Band):
        self.validate_type_hints()
        self.band = band

    def direct(self, input):
        if isinstance(self.band.where, dict):
            self._validate_coord_names(
                input,
                self.band.where,
                parameter="where",
                conditions_only=True,
            )

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

        return replace(
            band_spec,
            color=None,
            opacity=None,
            kwargs=new_kwargs,
        )

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
