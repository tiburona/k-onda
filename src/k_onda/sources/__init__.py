from .core import (
    Collection,
    DataComponent,
    DataIdentity,
    DataSource,
    CollectionMap,
    SignalMap
)

from .lfp_sources import LFPChannel, LFPRecording

__all__ = [
    "DataSource",
    "LFPRecording",
    "DataComponent",
    "DataIdentity",
    "LFPChannel",
    "Collection",
    "CollectionMap",
    "SignalMap"
]

try:
    from .spike_sources import (
        Neuron,
        PhyOutput,
        SpikeCluster,
        initialize_neurons_from_phy,
    )

    __all__.extend(
        ["PhyOutput", "Neuron", "SpikeCluster", "initialize_neurons_from_phy"]
    )
except ModuleNotFoundError:
    pass
