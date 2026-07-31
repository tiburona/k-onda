from .core import PlotSource


class SetAxisSharing():
    pass


class PlotRoleResolver:

    def resolve(self, plot_node, layout_spec):
        if plot_node.plot_type in ("histogram", "time-histogram", "psth"):
            return self.resolve_histogram(plot_node, layout_spec)
        raise NotImplementedError("Only histogram plots are currently implemented")

    def resolve_histogram(self, plot_node, layout_spec):
        return {
            "x": PlotSource(
                kind="coord",
                name=self.infer_x_source(plot_node, layout_spec),
            ),
            "y": PlotSource(kind="values"),
        }

    def infer_x_source(self, input, layout_spec):
        if input.coords:
            return next(iter(input.coords.values()))

        data_schema = input.data_source.data_schema
        panel_coord_names = set().union(
            *(panel.coords.keys() for panel in layout_spec.flat_panels)
        )
        default_axes = data_schema.axis_names_minus_axes_with_coords(
            panel_coord_names
        )

        if len(default_axes) != 1:
            raise ValueError("Unable to determine default axis.")
        return default_axes[0]
