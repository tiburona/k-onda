import matplotlib.pyplot as plt
from phylib.io.model import load_model
from pathlib import Path
import numpy as np
from scipy.signal import medfilt
from collections import defaultdict
import csv
import os


class PhyInterface:
    """An interface to data from Phy, the cluster visualization program.  Fetches principal component features and
    waveforms for plotting."""

    def __init__(self, path, animal):
        self.animal = animal
        self.path = Path(path)
        self.model = load_model(self.path / 'params.py')
        self.cluster_dict = self.read_curated_cluster_groups()
        self.spike_times = self.model.spike_times
        self.spike_clusters = np.load(self.path / 'spike_clusters.npy')
        if os.path.exists(self.path / f'{self.animal}.tsv'):
            self.peak_electrodes_file = self.path / f'{self.animal}.tsv'
        else:
            self.peak_electrodes_file = None
        self.assemble_cluster_dictionary()

    def read_curated_cluster_groups(self):
        # Open the curated cluster groups file (from Phy)
        with open(self.path / 'cluster_info.tsv') as file:
        # Use DictReader to read the file with headers
            tsv_file = csv.DictReader(file, delimiter='\t')

            # Extract relevant fields: 'cluster_id' and 'group'
            return {int(row['cluster_id']): {'animal': self.animal, 'group': row['group'], 
                                                  'cluster': int(row['cluster_id'])}
                for row in tsv_file if row['group'] != 'noise'}

    def assemble_cluster_dictionary(self):
        for cluster in self.cluster_dict:
            self.cluster_dict[cluster]['spike_times'] = self.get_spike_times_for_cluster(cluster)
        if self.peak_electrodes_file:
            for row in self.read_peak_electrodes_file():
                cluster = int(row['Cluster'])
                self.cluster_dict[cluster]['electrodes'] = [int(electrode) for electrode in row['Electrodes'].split(',')]
                self.cluster_dict[cluster]['deflection'] = 'max' if row['Deflection'] in ['max', 'up'] else 'min'
                self.cluster_dict[cluster]['quality'] = row['Quality']
        else:
            for cluster, data in self.cluster_dict.items():
                data['electrodes'] = [self.model.clusters_channels[cluster]]

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
        feature_vals = self.get_all_pairwise_feature_views(cluster_ids, electrodes)
        for row in range(4):
            for col in range(4):
                for i, id in enumerate(cluster_ids):
                    x, y = feature_vals[i][row][col]
                    axs[row, col].scatter(x, y, color=colors[i], alpha=0.5)
                    if row == col:
                        axs[row, col].set_xlabel('time')
                    else:
                        axs[row, col].set_xlabel(f"{electrodes[col % 2]}{['A', 'B'][col > 1]}")
                    axs[row, col].set_ylabel(f"{electrodes[row % 2]}{['A', 'B'][int(row > 1)]}")

        plt.show()

    def get_all_pairwise_feature_views(self, cluster_ids, electrodes):
        vals_to_return = defaultdict(lambda: [[] for _ in range(4)])
        for row in range(4):
            for col in range(4):
                if row == col:
                    for i, id in enumerate(cluster_ids):
                        times = self.get_spike_times_for_cluster(id)
                        features = self.get_features(id, electrodes)[:, row % 2, int(row > 1)]
                        vals_to_return[i][row].append((times, features))
                else:
                    pc_inds = (int(col > 1), int(row > 1))
                    el_inds = (col % 2, row % 2)
                    for i, id in enumerate(cluster_ids):
                        x, y = (self.one_feature_view(id, electrodes, el_inds, pc_inds))
                        vals_to_return[i][row].append((x, y))
        return vals_to_return

    def one_feature_view(self, id, electrodes, el_inds, pc_inds):
        return (self.get_features(id, electrodes)[:, el, pc] for el, pc in zip(el_inds, pc_inds))

    def get_mean_waveforms_on_peak_electrodes(self, cluster_id):
        electrodes = self.cluster_dict[cluster_id]['electrodes']
        return self.get_mean_waveforms(cluster_id, electrodes)

    def get_mean_waveforms(self, cluster_id, electrodes):
        channels_used = self.model.get_cluster_channels(cluster_id)
        indices = np.where(np.isin(channels_used, electrodes))[0]
        waveforms = self.model.get_cluster_spike_waveforms(cluster_id)
        filtered_waveforms = medfilt(waveforms, kernel_size=[1, 5, 1])
        averaged_waveforms = np.mean(filtered_waveforms[:, :, indices], axis=(0, 2))
        return averaged_waveforms

    def read_peak_electrodes_file(self):
        if not self.peak_electrodes_file:
            return
        with open(self.path / f'{self.animal}.tsv') as f:
            reader = csv.DictReader(f, delimiter='\t')
            return [row for row in reader]


#phy_interface = PhyInterface('/Users/katie/likhtik/IG_INED_Safety_Recall', 'IG156')
#phy_interface.plot_features([54, 57], [8, 7])

