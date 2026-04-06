from .core import (
    Collection,
    DataComponent,
    DataIdentity,
    DataSource,
    CollectionMap,
    SignalMap
)

from .lfp_sources import LFPChannel, LFPRecording, LFPBrainRegion

__all__ = [
    "DataSource",
    "LFPRecording",
    "LFPBrainRegion",
    "DataComponent",
    "DataIdentity",
    "LFPChannel",
    "Collection",
    "CollectionMap",
    "SignalMap"
]

from .spike_sources import (
        Neuron,
        PhyOutput,
        SpikeCluster
    )
   

__all__.extend(
        ["PhyOutput", "Neuron", "SpikeCluster"]
    )


class DataIdentityView:

    def __init__(self, data_identity, subject):
        self.data_identity = data_identity
        self.subject = subject
        self.attributes = {}