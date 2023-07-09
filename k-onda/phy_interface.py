import matplotlib.pyplot as plt
from phylib.io.model import load_model
from pathlib import Path
import numpy as np
from scipy.signal import medfilt

animal_path = Path('/Users/katie/likhtik/data/single_cell_data/IG158')
model = load_model(animal_path / 'params.py')
spike_times = np.load(animal_path / 'spike_times.npy')
spike_clusters = np.load(animal_path / 'spike_clusters.npy')


class PhyInterface:

    def __init__(self, path, animal):
        self.animal = animal
        path = Path(path)
        self.path = path / animal
        self.model = load_model(self.path / 'params.py')
        self.spike_times = self.model.spike_times
        self.spike_clusters = np.load(self.path / 'spike_clusters.npy')

    def get_spike_ids_for_cluster(self, cluster_id):
        spike_ids = np.where(self.spike_clusters == cluster_id)[0]
        return spike_ids.tolist()

    def get_features(self, cluster_id, electrodes):
        cluster_spike_ids = self.get_spike_ids_for_cluster(cluster_id)
        return self.model.get_features(cluster_spike_ids, electrodes)

    def get_spike_times_for_cluster(self, cluster_id):
        return self.spike_times[self.get_spike_ids_for_cluster(cluster_id)]

    def plot_features(self, cluster_ids, electrodes):
        colors = ['red', 'blue']
        fig, axs = plt.subplots(4, 4, figsize=(15, 15))
        for row in range(4):
            for col in range(4):
                if row == col:
                    for i, id in enumerate(cluster_ids):
                        times = self.get_spike_times_for_cluster(id)
                        features = self.get_features(id, electrodes)[:, row % 2, int(row > 1)]
                        axs[row, col].scatter(times, features, color=colors[i], alpha=0.5)
                    axs[row, col].legend()
                else:
                    pc_inds = (int(col > 1), int(row > 1))
                    el_inds = (col % 2, row % 2)
                    for i, id in enumerate(cluster_ids):
                        x, y = (self.one_feature_view(id, electrodes, el_inds, pc_inds))
                        axs[row, col].scatter(x, y, alpha=0.5, color=colors[i])
                    axs[row, col].legend()

        plt.show()

    def one_feature_view(self, id, electrodes, el_inds, pc_inds):
        return (self.get_features(id, electrodes)[:, el, pc] for el, pc in zip(el_inds, pc_inds))

    def get_mean_waveforms(self, cluster_id, electrodes):
        channels_used = self.model.get_cluster_channels(cluster_id)
        indices = np.where(np.isin(channels_used, electrodes))[0]
        waveforms = self.model.get_cluster_spike_waveforms(cluster_id)
        filtered_waveforms = medfilt(waveforms, kernel_size=[1, 5, 1])
        averaged_waveforms = np.mean(filtered_waveforms[::100, :, indices], axis=(0, 2))
        return averaged_waveforms


phy_interface = PhyInterface('/Users/katie/likhtik/data/single_cell_data', 'IG158')
# phy_interface.plot_features([33, 39], [9, 11])




