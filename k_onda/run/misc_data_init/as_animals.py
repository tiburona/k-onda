import os
import json

root = '/Users/katie/likhtik/AS'

lfp_electrodes = {'bla': 1, 'vhip': 0, 'pl': 2}
as110_lfp_electrodes = {'vhip': 0, 'pl': 3, 'bla':1}
as113_lfp_electrodes = {'bla': 2, 'vhip': 0, 'pl': 3}

male_stressed = ['As107', 'As108']
male_non_stressed = ['As105', 'As106']
female_stressed = ['As112', 'As113']
female_non_stressed = ['As110', 'As111']

animals_and_groups = zip(
    [male_stressed, male_non_stressed, female_stressed, female_non_stressed],
    ['male_stressed', 'male_non_stressed', 'female_stressed', 'female_non_stressed'],
    [{'sex': 'male', 'treatment': 'stressed'}, {'sex': 'male', 'treatment': 'non_stressed'},
     {'sex': 'female', 'treatment': 'stressed'}, {'sex': 'female', 'treatment': 'non_stressed'}])

animal_info = {}

for animals, group_name, conditions in animals_and_groups:
    for animal in animals:
        if animal == 'As113': 
            electrodes = as113_lfp_electrodes
        elif animal == 'As110':
            electrodes = as110_lfp_electrodes
        else:
            electrodes = lfp_electrodes
        animal_info[animal] = {'group_name': group_name, 'conditions': conditions, 'lfp_electrodes': electrodes}

period_info = {
    'pretone_plus': {'relative': True, 'target': 'cs_plus', 'shift': -30, 'duration': 30,
                     'conditions': {'tone': 'off', 'cs': 'plus'}},
    'pretone_minus': {'relative': True, 'target': 'cs_plus', 'shift': -30, 'duration': 30,
                      'conditions': {'tone': 'off', 'cs': 'minus'}},
    'cs_plus': {'relative': False, 'reference_period_type': 'pretone', 'event_duration': 1, 
             'duration': 30, 'events': {'start': 'period_onset', 'pattern': {'range_args': [30]}},
             'conditions': {'tone': 'on', 'cs': 'plus'},
             'nev': {'code': 65503, 'indices': [1, 3, 6, 7, 9, 10]}},
    'cs_minus': {'relative': False, 'reference_period_type': 'pretone', 'event_duration': 1, 
             'duration': 30, 'events': {'start': 'period_onset', 'pattern': {'range_args': [30]}},
             'conditions': {'tone': 'off', 'cs': 'minus'},
             'nev': {'code': 65503, 'indices': [0, 2, 4, 5, 8, 11]}}
}

for animal, info in animal_info.items():
    info['period_info'] = period_info

animal_info_list = [info | {'identifier': id} for id, info in animal_info.items()]

exp_info = {}

exp_info['animals'] = animal_info_list
exp_info['group_names'] = ['male_stressed', 'female_stressed', 'male_non_stressed', 
                          'female_non_stressed']
exp_info['sampling_rate'] = 30000
exp_info['lfp_sampling_rate'] = 2000
exp_info['frequency_bands'] = {
    'theta_1': (4, 8),
    'theta_2': (8, 12)
}
exp_info['identifier'] = 'AS'
exp_info['data_path'] = root
exp_info['path_constructors'] = {
    'nev' : 
        {'template': root + '/{identifier}/Testing.mat', 
         'fields': ['identifier']},
    'spike':
        {'template': root + '/spike/{identifier}', 
         'fields': ['identifier']},
    'lfp': 
        {'template': root + '/{identifier}/Testing', 
         'fields': ['identifier']}  
}


with open(os.path.join(root, 'init_config.json'), 'w', encoding='utf-8') as file:
    json.dump(exp_info, file, ensure_ascii=False, indent=4)