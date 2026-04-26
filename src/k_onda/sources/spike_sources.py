import csv

import numpy as np
import xarray as xr
import pint

from ..central import Schema, DatasetSchema, AxisInfo, AxisKind
from ..signals import PointProcessSignal
from .core import DataComponent, DataIdentity, DataSource
from k_onda.central import types



@types.register
class PhyOutput(DataSource):
    def __init__(self, session, data_loader_config, sampling_rate=None):
        super().__init__(session, data_loader_config)
        self._sampling_rate = sampling_rate
        self._raw_data = None
        self._phy_model = None
        self._spike_times = None
        self._cluster_groups = None
        self.spike_clusters = np.load(self.file_path / "spike_clusters.npy")
        self._components = []

    @property
    def sampling_rate(self):
        if self._sampling_rate is None:
            self._sampling_rate = self.phy_model.sample_rate * pint.application_registry.Hz
        return self._sampling_rate

    @property
    def phy_model(self):
        if self._phy_model is None:
            from phylib.io.model import load_model

            self._phy_model = load_model(self.file_path / "params.py")
            self._phy_model.n_samples_waveforms = 200
        return self._phy_model

    @property
    def spike_times(self):
        if self._spike_times is None:
            self._spike_times = self.phy_model.spike_times
        return self._spike_times

    @property
    def cluster_groups(self):
        if self._cluster_groups is None:
            self._cluster_groups = self.get_cluster_groups()
        return self._cluster_groups
    
    @property
    def components(self):
        if not len (self._components):
            for cluster_id, group in self.cluster_groups.items():
                if group == "good":
                    spike_cluster = SpikeCluster(self, cluster_id)
                    self._components.append(spike_cluster)
        return self._components
                
    def get_cluster_groups(self):
        with open(self.file_path / "cluster_info.tsv") as file:
            tsv_file = csv.DictReader(file, delimiter="\t")
            cluster_groups = {
                int(row["cluster_id"]): row["group"]
                for row in tsv_file
                if row["group"] != "noise"
            }
            return cluster_groups

    def get_waveforms(self, cluster_idx):
        electrodes = [self.phy_model.clusters_channels[cluster_idx]]
        channels_used = self.phy_model.get_cluster_channels(cluster_idx)
        indices = np.where(np.isin(channels_used, electrodes))[0]
        waveforms = self.phy_model.get_cluster_spike_waveforms(cluster_idx)
        selected_waveforms = waveforms[:, :, indices]
        return selected_waveforms

    def get_spike_ids_for_cluster(self, cluster_id):
        spike_ids = np.where(self.spike_clusters == cluster_id)[0]
        return spike_ids.tolist()

    def get_features(self, cluster_id, electrodes):
        cluster_spike_ids = self.get_spike_ids_for_cluster(cluster_id)
        return self.model.get_features(cluster_spike_ids, electrodes)

    def get_spike_times_for_cluster(self, cluster_idx):
        return self.spike_times[self.get_spike_ids_for_cluster(cluster_idx)]
    
    
@types.register
class Neuron(DataIdentity):
    name = 'neuron'
    _snapshot_fields = DataIdentity._snapshot_fields + ('neuron_type',)

    def __init__(self, data_components, config):
        super().__init__(data_components)
        self.neuron_type = None
        self._label = self.generate_label()

    def generate_label(self):
        preexisting_neurons = self.subject.data_identities['neurons']
        integer_ids = [int(neuron.label[7:]) for neuron in preexisting_neurons]
        integer_id = 0 if not len(integer_ids) else max(integer_ids) + 1
        label = f'neuron{integer_id}'
        return label

    @property
    def label(self):
        return self._label
    
@types.register
class SpikeCluster(DataComponent):
    output_class = PointProcessSignal
    data_type = xr.Dataset

    def __init__(self, data_source, cluster_id, neuron=None):
        super().__init__(data_source)
        self.cluster_id = cluster_id
        self.component_id = (self.data_source)
        if neuron is not None:
            self.assign_to_neuron(neuron)

    @property
    def identifiers(self):
        return [f"cluster_{self.cluster_id}"]
    
    @property
    def data_schema(self):
        spike_times_schema = Schema(
            axes=[AxisInfo('spikes', kind=AxisKind.POINT_PROCESS_INDEX)],
            value_metadim='time')
        waveforms_schema = Schema(
            axes=[AxisInfo('spikes', AxisKind.POINT_PROCESS_INDEX, metadim=None),
                  AxisInfo('samples', AxisKind.AXIS, metadim='time'),
                  AxisInfo('electrodes', AxisKind.AXIS, metadim=None)],
            value_metadim='voltage'
        )
        return DatasetSchema({
             'spike_times': spike_times_schema,
             'waveforms': waveforms_schema
        })

    @property
    def neuron(self):
        return self.data_identity
    
    def to_signal(self):
        signal = super().to_signal()
        signal.duration = self.duration
        return signal


    def data_loader(self):
        spike_times = self.data_source.get_spike_times_for_cluster(self.cluster_id)
        spike_times = xr.DataArray(
            spike_times * pint.application_registry.s,  # TODO is this the actual unit they come from phy in?
            dims=("spikes",),
        )
        waveforms = self.data_source.get_waveforms(self.cluster_id)
        waveforms = xr.DataArray(waveforms, dims=('spikes', 'samples', 'electrodes'))

        return xr.Dataset(
            {'spike_times': spike_times, 'waveforms': waveforms}
        )

    def assign_to_neuron(self, neuron):
        self.assign_to_data_identity(neuron)



