
from phy_interface import PhyInterface
from math_functions import full_width_half_minimum

ROOT_DIR = r"D:\back_up_lenovo\data\Single_Cell_Data_No_Uv"
STANDARD_ANIMALS = ['IG160', 'IG163', 'IG176', 'IG178', 'IG180', 'IG154', 'IG156', 'IG158', 'IG177', 'IG179']


for animal in STANDARD_ANIMALS:
    phy_interface = PhyInterface(ROOT_DIR, animal)

    units_info = {'good': [], 'MUA': []}
    for cluster in phy_interface.spike_groups:
        if phy_interface.spike_groups[cluster] in ['good', 'MUA']:
            spike_times = phy_interface.get_spike_times_for_cluster(cluster)
            mean_waveform = phy_interface.get_mean_waveforms(cluster, phy_interface.get_peak_electrodes(cluster))
            fwhm = full_width_half_minimum(mean_waveform, 30000)





