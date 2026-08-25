from k_onda.mixins import DictDelegator


class FeatureRegistry(DictDelegator):
    _delegate_attr = "registry"

    def __init__(self):
        self.registry = {}


feature_registry = FeatureRegistry()


def fwhm(
    input,
    *,
    fwhm_dim="sample",
    reduce_dim="spike",
    include_valleys=True,
    peak_selection="prominence",
    key="waveforms",
    key_output_mode="standalone",
):
    return (
        input.fwhm(
            dim=fwhm_dim,
            include_valleys=include_valleys,
            peak_selection=peak_selection,
            key=key,
            key_output_mode=key_output_mode,
        )
        .reduce(reduce_dim)
        .mean()
    )


def firing_rate(
    input,
    *,
    intervals=None,
    exclude_initial=None,
    key=None,
    key_output_mode=None,
):
    return input.rate(
        intervals=intervals,
        exclude_initial=exclude_initial,
        key=key,
        key_output_mode=key_output_mode,
    ).mean()


feature_registry["fwhm"] = fwhm
feature_registry["firing_rate"] = firing_rate
