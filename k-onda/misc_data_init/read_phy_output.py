from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from neo.rawio import BlackrockRawIO
from scipy.spatial.distance import mahalanobis


from phy_interface import PhyInterface
from math_functions import get_fwhm

ROOT_DIR = r"D:\back_up_lenovo\data\Single_Cell_Data_No_Uv_Diff_Scale"
STANDARD_ANIMALS = ['IG160', 'IG163', 'IG176', 'IG178', 'IG180', 'IG154', 'IG156', 'IG158', 'IG177', 'IG179']
SAMPLING_RATE = 30000

animals = {}
all_good_units = []


def calculate_mean_waveform(animal, spike_times, electrodes):
    reader = BlackrockRawIO(filename=os.path.join(ROOT_DIR, animal, 'Safety'), nsx_to_load=6)
    reader.parse_header()
    data = np.array([reader.nsx_datas[6][0][:, electrode] for electrode in electrodes])
    average_data = np.mean(data, 0)
    spike_times = [int(spike_time*SAMPLING_RATE) for spike_time in spike_times]
    waveforms = np.array([average_data[spike_time-41:spike_time+41] for spike_time in spike_times
                          if 41 < spike_time < len(average_data) - 41])
    mean_waveform = np.mean(waveforms, 0)
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




def categorize_neurons(units, fwhm_max=.0015, firing_rate_max=10):

    # Separate units with quality 1 for initial categorization
    quality_1_units = [unit for unit in units if int(unit['quality']) == 1]

    # Categorize quality 1 units based on FWHM and firing rate
    fat_units = [unit for unit in quality_1_units if unit['fwhm'] > fwhm_max]
    fast_units = [unit for unit in quality_1_units if unit['firing_rate'] > firing_rate_max]
    for unit in fat_units:
        unit['neuron_type'] = 'PN'
    for unit in fast_units:
        unit['neuron_type'] = 'IN'

    # Normalize FWHM and firing rate for quality 1 units
    normal_units = [unit for unit in quality_1_units if unit not in fat_units and unit not in fast_units]

    firing_rates = [unit['firing_rate'] for unit in normal_units]
    fwhm = [unit['fwhm'] for unit in normal_units]
    reshaped_firing_rates = np.array(firing_rates).reshape(-1, 1)
    reshaped_fwhm = np.array(fwhm).reshape(-1, 1)
    scaler = StandardScaler()
    firing_rates_transformed = scaler.fit_transform(reshaped_firing_rates)
    fwhm_transformed = scaler.fit_transform(reshaped_fwhm)
    X = np.column_stack([firing_rates_transformed, fwhm_transformed])

    # Perform KMeans clustering for quality 1 units
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    highest_center_index = np.argmax(centers[:, 0])
    # The label of this cluster is the same as the index
    IN_label = highest_center_index

    for unit, label in zip(normal_units, labels):
        unit['neuron_type'] = 'IN' if label == IN_label else 'PN'

    # Separate quality 2 and 3 units
    quality_23_units = [unit for unit in units if unit not in quality_1_units]
    firing_rates_quality_23 = [unit['firing_rate'] for unit in quality_23_units]
    fwhm_quality_23 = [unit['fwhm'] for unit in quality_23_units]

    all_firing_rates = [unit['firing_rate'] for unit in units]
    all_fwhm = [unit['fwhm'] for unit in units]
    all_firing_rates_scaled = scaler.fit_transform(np.array(all_firing_rates).reshape(-1, 1))
    all_fwhm_scaled = scaler.fit_transform(np.array(all_fwhm).reshape(-1, 1))

    # Separate scaled values for quality 1 units
    quality_1_indices = [i for i, unit in enumerate(units) if int(unit['quality']) == 1]
    quality_1_firing_rates_scaled = all_firing_rates_scaled[quality_1_indices]
    quality_1_fwhms_scaled = all_fwhm_scaled[quality_1_indices]

    # Separate scaled values for quality 2 or 3 units
    quality_23_indices = [i for i, unit in enumerate(units) if int(unit['quality']) > 1]
    quality_23_firing_rates_scaled = all_firing_rates_scaled[quality_23_indices]
    quality_23_fwhms_scaled = all_fwhm_scaled[quality_23_indices]

    for unit, firing_rate_scaled, fwhm_scaled in zip(quality_23_units, quality_23_firing_rates_scaled, quality_23_fwhms_scaled):
        unit_features = np.array([firing_rate_scaled[0], fwhm_scaled[0]])
        distance = float('inf')
        for categorized_unit, quality_1_firing_rate_scaled, quality_1_fwhm_scaled in zip(
                quality_1_units, quality_1_firing_rates_scaled, quality_1_fwhms_scaled):
            categorized_unit_features = np.array([quality_1_firing_rate_scaled[0], quality_1_fwhm_scaled[0]])
            test_distance = np.linalg.norm(unit_features - categorized_unit_features)
            if test_distance < distance:
                distance = test_distance
                best_group = categorized_unit['neuron_type']
        unit['neuron_type'] = best_group

    return units


def remove_np_arrays(unit):
    for k, v in unit.items():
        if isinstance(v, np.ndarray):
            unit[k] = v.tolist()


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
    for unit in all_good_units:
        remove_np_arrays(unit)
    for animal_name in animal_dict:
        path = os.path.join(ROOT_DIR, animal_name)
        with open(os.path.join(path, f'{animal_name}_units_info.json'), 'w') as json_file:
            json.dump(animal_dict[animal_name], json_file)


categorize_neurons(all_good_units)
show_scatterplot([unit for unit in all_good_units if int(unit['quality']) == 1])
show_scatterplot(all_good_units)
save_units_info(animals, check_with_user=False)









