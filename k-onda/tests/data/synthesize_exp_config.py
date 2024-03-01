import numpy as np
import json

# Configuration parameters
animals = ['TESTMOUSE1', 'TESTMOUSE2', 'TESTMOUSE3', 'TESTMOUSE4']
conditions = ['control', 'control', 'stress', 'stress']
neuron_types = ['IN', 'PN']
period_types = ['pretone', 'tone']
sampling_rate = 1000  # Hz, for converting times to samples
pretone_duration = 30  # seconds
tone_duration = 30  # seconds
pretone_firing_rate = 5  # spikes per second
tone_firing_rate = 10  # spikes per second
num_periods = 2  # Number of pretone and tone periods



# Helper function to generate spike times
def generate_spike_times(start_time, duration, firing_rate, sampling_rate):
    total_spikes = int(duration * firing_rate)
    spike_times = np.sort(np.random.randint(start_time, start_time + duration * sampling_rate, total_spikes))
    return spike_times.tolist()

def generate_alternating_period_spike_times(pretone_duration, tone_duration, pretone_firing_rate, tone_firing_rate, sampling_rate):
    spike_times = []
    current_start_time = 0  # Start at the beginning

    for _ in range(num_periods):
        # Generate pretone spikes
        pretone_spikes = generate_spike_times(current_start_time, pretone_duration, pretone_firing_rate, sampling_rate)
        spike_times.extend(pretone_spikes)
        current_start_time += pretone_duration * sampling_rate

        # Generate tone spikes
        tone_spikes = generate_spike_times(current_start_time, tone_duration, tone_firing_rate, sampling_rate)
        spike_times.extend(tone_spikes)
        current_start_time += tone_duration * sampling_rate

    return spike_times


# Generate configuration
exp_config = {'identifier': 'TEST_EXPERIMENT', 'conditions': ['stress', 'control'], 'neuron_types': ['PN', 'IN'],
          'animals': [], 'sampling_rate': sampling_rate}


for animal_id, condition in zip(animals, conditions):
    animal_config = {
        'identifier': animal_id,
        'condition': condition,
        'period_info': {
            'pretone': {
                'relative': True,
                'target': 'tone',
                'shift': -30,
                'duration': pretone_duration
            },
            'tone': {
                'onsets': [pretone_duration * sampling_rate, pretone_duration * sampling_rate * 2],
                # Assuming two tones starting at 0 and after the first pretone
                'events': [[30000 + i * sampling_rate for i in range(30)], [60000 + i * sampling_rate for i in range(30)]],
                'duration': tone_duration,
                'event_duration': 1,
                'reference_period_type': 'pretone'
            }
        },
        'units': {'good': []}
    }

    # Generate units with spike times
    for _ in range(2):  # Two units of each type per animal
        for neuron_type in neuron_types:
            spike_times = generate_alternating_period_spike_times(pretone_duration, tone_duration, pretone_firing_rate,
                                                                  tone_firing_rate, sampling_rate)
            unit = {
                'neuron_type': neuron_type,
                'spike_times': spike_times
            }
            animal_config['units']['good'].append(unit)

    exp_config['animals'].append(animal_config)

with open('./exp_config.json', 'w') as file:
    json.dump(exp_config, file, indent=4)



