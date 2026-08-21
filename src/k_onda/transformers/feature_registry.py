from k_onda.mixins import DictDelegator


class FeatureRegistry(DictDelegator):
    _delegate_attr = "registry"

    def __init__(self):
        self.registry = {}


feature_registry = FeatureRegistry()


def fwhm(
        input, 
        fwhm_dim="samples", 
        reduce_dim="spikes",
        include_valleys=True, 
        permissible_distance=75, 
        distance_unit=None, 
        key="waveforms",
        key_output_mode="standalone"
        ):
    
    return (
        input.fwhm(
            dim=fwhm_dim, 
            include_valleys=include_valleys, 
            permissible_distance=permissible_distance, 
            distance_unit=distance_unit, 
            key=key,
            key_output_mode=key_output_mode)
        .reduce(reduce_dim)
        .mean()
    )


def firing_rate(input, config=None):
    if config is None:
        config = {}

    return input.rate(**config).mean()


feature_registry["fwhm"] = fwhm
feature_registry["firing_rate"] = firing_rate
