from k_onda.mixins import DictDelegator


class FeatureRegistry(DictDelegator):
    _delegate_attr = "registry"

    def __init__(self):
        self.registry = {}


feature_registry = FeatureRegistry()


def fwhm(input, config=None):
    if config is None:
        config = {}

    key = config.pop("key", "waveforms")
    dim = config.pop("dim", "spikes")

    return (
        input.fwhm(**config, key=key, key_output_mode="standalone")
        .reduce(dim)
        .aggregate()
    )


def firing_rate(input, config=None):
    if config is None:
        config = {}

    return input.rate(**config).aggregate()


feature_registry["fwhm"] = fwhm
feature_registry["firing_rate"] = firing_rate
