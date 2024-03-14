from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from neo.rawio import BlackrockRawIO

from phy_interface import PhyInterface
from math_functions import get_fwhm

ROOT_DIR = r"D:\back_up_lenovo\data\Single_Cell_Data_No_Uv_Diff_Scale"
STANDARD_ANIMALS = ['IG160', 'IG163', 'IG176', 'IG178', 'IG180', 'IG154', 'IG156', 'IG158', 'IG177', 'IG179']
SAMPLING_RATE = 30000

animals = {}
all_good_units = []

def calculate_mean_waveform(animal, spike_times, electrodes):
    reader = BlackrockRawIO(filename=os.path.join(ROOT_DIR, animal), nsx_to_load=6)
    reader.parse_header()
    data = reader.nsx_datas[6][0][:, slice(*electrodes)]
    average_data = np.mean(data, 1)
    spike_times = [int(spike_time*SAMPLING_RATE) for spike_time in spike_times]
    waveforms = np.array([data[spike_time-41:spike_time+41] for spike_times in spike_times])
    mean_waveform = np.mean(waveforms, 1)
    return mean_waveform


for animal in STANDARD_ANIMALS:
    phy_interface = PhyInterface(ROOT_DIR, animal)
    units_info = {'good': [], 'MUA': []}
    for cluster in phy_interface.cluster_dict:
        cluster_type = phy_interface.cluster_dict[cluster]['group']
        if cluster_type in ['good', 'MUA']:
            spike_times = phy_interface.get_spike_times_for_cluster(cluster)
            electrodes = phy_interface.cluster_dict[cluster]['electrodes']
            mean_waveform = calculate_mean_waveform(animal, spike_times, electrodes)
            deflection = phy_interface.cluster_dict[cluster]['deflection']
            if deflection == 'max':
                range_of_max = (-25, 25)
                range_of_min = (0, 35)
            else:
                range_of_max = (0, 35)
                range_of_min = (-25, 25)
            fwhm = get_fwhm(mean_waveform, SAMPLING_RATE, deflection=deflection)
            usable_spike_times = spike_times[spike_times > 100]
            firing_rate = len(usable_spike_times)/(usable_spike_times[-1] - usable_spike_times[0])
            unit_info = {**phy_interface.cluster_dict[cluster],
                         **{'mean_waveform': list(mean_waveform), 'firing_rate': firing_rate, 'fwhm': fwhm}}
            units_info[cluster_type].append(unit_info)
            if cluster_type == 'good':
                all_good_units.append(unit_info)
            animals[animal] = units_info


def categorize_neurons(units):
    firing_rates = [unit['firing_rate'] for unit in units]
    fwhm = [unit['fwhm'] for unit in all_good_units]
    scaler = StandardScaler()
    firing_rates = scaler.fit_transform(np.array(firing_rates).reshape(-1, 1))
    fwhm = scaler.fit_transform(np.array(fwhm).reshape(-1, 1))
    X = np.column_stack([firing_rates, fwhm])

    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    highest_center_index = np.argmax(centers[:, 0])
    # The label of this cluster is the same as the index
    IN_label = highest_center_index
    for unit, label in zip(units, labels):
        unit['neuron_type'] = 'IN' if label == IN_label else 'PN'


def show_scatterplot(units):
    fwhm_IN = [unit['fwhm'] * 1000000 for unit in units if unit['neuron_type'] == 'IN']
    firing_rate_IN = [unit['firing_rate'] for unit in units if unit['neuron_type'] == 'IN']

    fwhm_PN = [unit['fwhm'] * 1000000 for unit in units if unit['neuron_type'] == 'PN']
    firing_rate_PN = [unit['firing_rate'] for unit in units if unit['neuron_type'] == 'PN']

    plt.figure(figsize=(10, 6))

    plt.scatter(fwhm_IN, firing_rate_IN, color='blue', label='IN')
    plt.scatter(fwhm_PN, firing_rate_PN, color='red', label='PN')

    plt.title('FWHM vs Firing Rate by Neuron Type')
    plt.xlabel('FWHM (microseconds)')
    plt.ylabel('Firing Rate')

    plt.legend()
    plt.show()


def save_units_info(animal_dict, check_with_user=False):
    if check_with_user:
        user_input = input("Proceed with saving units info? (y/n): ")
        if user_input.lower() != 'y':
            print("Operation aborted.")
            return

    for animal_name in animal_dict:
        path = os.path.join(ROOT_DIR, animal_name)
        with open(os.path.join(path, 'units_info.json'), 'w') as json_file:
            json.dump(animal_dict[animal_name], json_file)


categorize_neurons(all_good_units)
show_scatterplot(all_good_units)
save_units_info(animals, check_with_user=True)









