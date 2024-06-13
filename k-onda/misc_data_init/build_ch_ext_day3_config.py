import json
import os

root = '/Users/katie/likhtik/CH_EXT'

ANIMALS = ['CH054', 'CH069', 'CH129', 'CH130', 'CH131', 'CH134', 'CH135', 'CH151', 'CH152', 'CH154']
lfp_electrodes = {'bla': 8, 'bf': 10, 'il': 12} # TODO check this make sure I'm converting to 0 index correctlly

animal_info = {animal: {'condition': 'all', 'lfp_electrodes': lfp_electrodes} for animal in ANIMALS} # TODO if/when I find out conditions change this

exp_info = {
    'conditions': ['all'],
    'identifier': 'CH_EXT',
    'neuron_types': [],
    'sampling_rate': 30000,
    'lfp_sampling_rate': 2000,
    'lfp_root': root,
    'lfp_path_constructor': ['EXT'],
    'lost_signal': .5,
    'stimulus_duration': .05,
    'frequency_bands': dict(delta=(0, 4), theta_1=(4, 8), theta_2=(8, 12), delta_theta=(0, 12), gamma=(20, 55),
                            hgamma=(70, 120)),
    'behavior_data': '',
    'behavior_animal_id_column': ''
}

animals = []

for animal in ANIMALS:
    animal_file = os.path.join(root, animal, 'EXT.json')
    tone_on_code = 65502
    with open(animal_file, 'r', encoding='utf-8') as file:
        print(animal_file)
        data = file.read()
        json_data = json.loads(data)
        time_stamps = json_data['NEV']['Data']['SerialDigitalIO']['TimeStamp']
        unparsed_data = json_data['NEV']['Data']['SerialDigitalIO']['UnparsedData']
        tone_onsets = [ts for i, ts in enumerate(time_stamps) if unparsed_data[i] == tone_on_code]
        events = [[onset + i * 30000 for i in range(30)] for onset in tone_onsets]
        pretone = {'relative': True, 'target': 'tone', 'shift': -30, 'duration': 30, 'lfp_padding': [1, 1]}
        tone = {'onsets': tone_onsets, 'events': events, 'duration': 30, 'lfp_padding': [1, 1],
                'event_duration': 1, 'reference_period_type': 'pretone'}
        animals.append({'identifier': animal, 'period_info': {'tone': tone, 'pretone': pretone}, **animal_info[animal]})

exp_info['animals'] = animals

with open(os.path.join(root, 'init_config.json'), 'w', encoding='utf-8') as file:
    json.dump(exp_info, file, ensure_ascii=False, indent=4)








