import matplotlib.pyplot as plt
import os
import numpy as np

from k_onda.utils import save, load, PrepMethods
from k_onda.interfaces import PhyInterface
from .data_structures import Unit


class SpikePrepMethods(PrepMethods):

    def spike_prep(self):
        if 'spike' in self.initialized:
            return
        self.initialized.append('spike')
        self.units_prep()
        for unit in [unit for _, units in self.units.items() for unit in units]:
            unit.spike_prep()
        
    def select_spike_children(self):
        if self.selected_neuron_type:
            return self.neurons[self.selected_neuron_type]
        else: 
            return [unit for units in self.units.values() for unit in units]
        
    def units_prep(self):
        units_info = self.animal_info.get('units', {})
        if 'get_units_from_phy' in units_info.get('instructions', []):
            self.get_units_from_phy()

        for category in [cat for cat in ['good', 'MUA'] if cat in units_info]:
            for unit_info in units_info[category]:
                unit_kwargs = {kw: unit_info[kw] for kw in ['waveform', 'neuron_type', 'quality', 'firing_rate', 'fwhm']}
                unit = Unit(self, category, unit_info['spike_times'], unit_info['cluster'], 
                            experiment=self.experiment, **unit_kwargs)
                if unit.neuron_type:
                    getattr(self, unit.neuron_type).append(unit)

    def get_units_from_phy(self):
        saved_calc_exists, units, pickle_path = load(
            os.path.join(self.construct_path('spike'), 'units_from_phy.pkl'), 'pkl'
        )
        
        if saved_calc_exists:
            for unit in units:
                Unit(self, unit['group'], unit['spike_times'], unit['cluster'], 
                    waveform=unit['waveform'], experiment=self.experiment, 
                    firing_rate=unit['firing_rate'], fwhm=unit['fwhm'])
        else:
            phy_path = self.construct_path('phy')
            phy_interface = PhyInterface(phy_path, self)
            cluster_dict = phy_interface.cluster_dict
            units = []

            for cluster, info in cluster_dict.items():
                if info['group'] in ['good', 'mua']:
                    try:
                        waveform = phy_interface.get_mean_waveforms_on_peak_electrodes(cluster)
                    except ValueError as e:
                        if "argmax of an empty sequence" in str(e):
                            continue
                        else:
                            raise
                    
                    # Check if deflection is up or down at the center
                    center_value = waveform[100]
                    surrounding_mean = np.mean(np.concatenate((waveform[0:50], waveform[-50:])))
                    threshold = (np.max(waveform[70:130]) + np.min(waveform[70:130])) / 2

                    if center_value > surrounding_mean:  # Upward deflection
                        above_threshold = waveform >= threshold
                    else:  # Downward deflection
                        above_threshold = waveform <= threshold

                    # Find transitions (crossings)
                    transitions = np.diff(above_threshold.astype(int))

                    # Last False -> True transition before center (rising edge)
                    before_center = np.where(transitions[:101] == 1)[0]
                    start = before_center[-1] if len(before_center) > 0 else 70  # fallback start

                    # First True -> False transition after center (falling edge)
                    after_center = np.where(transitions[100:] == -1)[0] + 100
                    end = after_center[0] if len(after_center) > 0 else 130  # fallback end

                    # Calculate FWHM in samples and time
                    FWHM_samples = end - start
                    FWHM_seconds = FWHM_samples / self.sampling_rate

                    # Plot waveform and mark FWHM
                    plt.figure(figsize=(10, 5))
                    plt.plot(waveform, label="Waveform")
                    
                    # Horizontal line for half maximum level
                    plt.axhline(y=threshold, color='r', linestyle='--', label="Half Maximum")

                    # Mark FWHM region
                    plt.axvspan(start, end, color='orange', alpha=0.3, label="FWHM Range")
                    
                    plt.xlabel("Sample Index")
                    plt.ylabel("Amplitude")
                    plt.title(f"Waveform with FWHM for Animal {self.identifier} Cluster {cluster}")
                    plt.legend()
                    plt.savefig(os.path.join(phy_path, f"{self.identifier}_{cluster}_waveform.png"))
                    
                    # Firing rate calculation
                    spike_times = np.array(info['spike_times'])
                    spike_times_for_fr = spike_times[spike_times > 30]
                    firing_rate = len(spike_times_for_fr) / (spike_times_for_fr[-1] - spike_times_for_fr[0])

                    # Create unit and add attributes
                    unit = Unit(self, info['group'], info['spike_times'], cluster,
                                waveform=waveform, experiment=self.experiment, firing_rate=firing_rate, 
                                fwhm_seconds=FWHM_seconds)

                    # Append to units list
                    units.append(info | {'waveform': waveform, 'firing_rate': firing_rate, 'fwhm_seconds': FWHM_seconds})
            
            save(units, pickle_path, 'pkl')

