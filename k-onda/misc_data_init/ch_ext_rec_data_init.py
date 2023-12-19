import json
import os


ext_rec_path = '/Users/katie/likhtik/CH_for_katie_less_conservative'
json_from_matlab = 'single_cell_data.json'


with open('/Users/katie/likhtik/CH_for_katie_less_conservative/single_cell_data.json', 'r', encoding='utf-8') as file:
    data = file.read()
    json_data = json.loads(data)

neuron_classification_rule = dict(column_name='cluster_assignment', classifications={'PV_IN': [3], 'ACH': [1, 2]})

exp_info = dict(conditions=['arch', 'control'], identifier='CH_EXTREC', sampling_rate=30000,
                neuron_types=['ACH', 'PV_IN'], neuron_classification_rule=neuron_classification_rule,
                lfp_sampling_rate=2000, lfp_root='/Users/katie/likhtik/CH_for_katie_less_conservative',
                lfp_path_constructor=['EXTREC'], lfp_electrodes={'bla': 0, 'il': 2},
                lfp_from_stereotrodes={'nsx_num': 5, 'electrodes': {'bf':  [0, 1]}}, lost_signal=.5)

animals = []

for animal in json_data:
    animal_info = {}
    block_info = {'tone': {}, 'pretone': {}}
    animal_info['identifier'] = animal['animal']
    animal_info['condition'] = animal['group']
    block_info['tone']['onsets'] = animal['tone_period_onsets']
    block_info['tone']['events'] = [[onset + i*30000 for i in range(30)] for onset in animal['tone_period_onsets']]
    block_info['tone']['duration'] = 30
    block_info['tone']['event_duration'] = 1
    block_info['tone']['lfp_padding'] = [1, 1]
    block_info['pretone']['reference'] = True
    block_info['pretone']['target'] = 'tone'
    block_info['pretone']['shift'] = 30  # TODO: sometimes we might want reference blocks that are forward in time; this would clearer if it were negative here and added rather than subtracted later
    block_info['pretone']['duration'] = 30
    block_info['pretone']['lfp_padding'] = [1, 1]
    animal_info['block_info'] = block_info
    animal_info['units'] = animal['units']
    animals.append(animal_info)

exp_info['animals'] = animals

with open(os.path.join(ext_rec_path, 'init_config.json'), 'w', encoding='utf-8') as file:
    json.dump(exp_info, file, ensure_ascii=False, indent=4)


