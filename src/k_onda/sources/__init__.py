from .lfp_sources import LFPChannel, LFPRecording, LFPBrainRegion
from .core import (
    Collection,
    CollectionMap,
    DataComponent,
    DataIdentity,
    DataSource,
    SignalMap,
)
from .spike_sources import (
    Neuron,
    PhyOutput,
    SpikeCluster,
)

__all__ = [
    "DataSource",
    "LFPRecording",
    "LFPBrainRegion",
    "DataComponent",
    "DataIdentity",
    "LFPChannel",
    "Collection",
    "CollectionMap",
    "SignalMap",
    "PhyOutput",
    "Neuron",
    "SpikeCluster",
]


class DataIdentityView:
    def __init__(self, data_identity, subject):
        self.data_identity = data_identity
        self.subject = subject
        self.attributes = {}
