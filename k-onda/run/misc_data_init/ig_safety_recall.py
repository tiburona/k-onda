import os
import json
from copy import deepcopy

root = '/Users/katie/likhtik/IG_INED_SAFETY_RECALL'

ined_lfp_electrodes = {'bla': 1, 'bf': 2, 'pl': 3}

ig_electrodes = {'hpc': 0, 'bla': 2}

no_st_electrodes = {'hpc': 0, 'bla': 1, 'pl': 3}

pl_electrodes = {
    'IG154': (4, 6), 'IG155': (12, 14), 'IG156': (12, 14), 'IG158': (7, 14), 'IG160': (1, 8), 
    'IG161': (9, 11), 'IG162': (13, 3), 'IG163': (14, 8), 'IG175': (15, 4), 'IG176': (11, 12), 
    'IG177': (15, 4), 'IG178': (4, 6), 'IG179': (13, 15), 'IG180': (15, 4)
}

control_just_behavior = ['INED02', 'INED03', 'INED15', 'INED19', 'INED20', 'IG159']
defeat_just_behavior = ['INED08', 'INED10', 'INED13', 'INED34', 'INED35']

control_no_brain_dict, defeat_no_brain_dict = ({
    animal: {'condition': condition} for animal in animals} 
    for condition, animals in [('control', control_just_behavior), ('defeat', defeat_just_behavior)]
)

control_ined = ['INED18', 'INED17', 'INED16', 'INED05', 'INED04']
defeat_ined = ['INED01', 'INED06', 'INED07', 'INED09', 'INED11', 'INED12']

control_ined_dict = {animal: {'lfp_electrodes': ined_lfp_electrodes, 'condition': 'control'} 
                     for animal in control_ined}
defeat_ined_dict = {animal: {'lfp_electrodes': ined_lfp_electrodes, 'condition': 'defeat'} 
                    for animal in defeat_ined}

defeat_ig_st = ['IG154', 'IG155', 'IG156', 'IG158', 'IG175', 'IG177', 'IG179']
control_ig_st = ['IG160', 'IG161', 'IG162', 'IG163', 'IG176', 'IG178', 'IG180']

animals_with_units = ['IG160', 'IG163', 'IG176', 'IG178', 'IG180', 'IG154', 'IG156', 'IG158', 
                      'IG177', 'IG179']

control_ig_dict, defeat_ig_dict = ({
    animal: {'condition': condition, 'lfp_electrodes': ig_electrodes,
             'lfp_from_stereotrodes': {'nsx_num': 6, 'electrodes': {'pl': pl_electrodes[animal]}}}
    for animal in animals} 
    for condition, animals in [('control', control_ig_st), ('defeat', defeat_ig_st)])

for d in control_ig_dict, defeat_ig_dict:
    for animal in d:
        if animal in animals_with_units:
            d[animal]['units'] = {'instructions': ['get_units_from_phy']}


control_ig_no_st = ['IG171', 'IG173']
defeat_ig_no_st = ['IG172', 'IG174']

control_ig_no_st_dict = {animal: {'condition': 'control', 'lfp_electrodes': no_st_electrodes} 
                         for animal in control_ig_no_st}
defeat_ig_no_st_dict = {animal: {'condition': 'defeat', 'lfp_electrodes': no_st_electrodes} 
                        for animal in defeat_ig_no_st}

animal_info = {**control_ined_dict, **defeat_ined_dict, **control_ig_dict, **defeat_ig_dict, 
               **control_ig_no_st_dict, **defeat_ig_no_st_dict, **control_no_brain_dict, 
               **defeat_no_brain_dict}

mice_with_alt_code = ['IG171', 'IG172', 'IG173', 'IG174', 'IG175', 'IG176', 'IG177', 'IG178', 
                      'IG179', 'IG180']


period_info = {
    'pretone': {'relative': True, 'target': 'tone', 'shift': -30, 'duration': 30},
    'tone': {'relative': False, 'reference_period_type': 'pretone', 'event_duration': 1, 
             'duration': 30, 'events': {'start': 'period_onset', 'pattern': {'range_args': [30]}}}
}

for animal, info in animal_info.items():
    tone_on_code = 65502 if animal not in mice_with_alt_code else 65436
    period_info['tone']['code'] = tone_on_code
    info['period_info'] = deepcopy(period_info)
    if animal not in control_just_behavior + defeat_just_behavior:
        info['period_info']['instructions'] = ['periods_from_nev']

animal_info_list = [info | {'identifier': id} for id, info in animal_info.items()]

exp_info = {}

exp_info['animals'] = animal_info_list
exp_info['conditions'] = ['control', 'defeat']
exp_info['sampling_rate'] = 30000
exp_info['lfp_sampling_rate'] = 2000
exp_info['frequency_bands'] = {
    'theta_1': (0, 4),
    'theta_2': (8, 12)
}
exp_info['identifier'] = 'IG_SAFETY_RECALL'
exp_info['path_constructors'] = {
    'nev' : 
        {'template': '/Users/katie/likhtik/IG_INED_Safety_Recall/{identifier}/{identifier}.mat', 'fields': ['identifier']},
    'phy': 
        {'template': '/Volumes/SanDisk/single_cell_data_ks_22/{identifier}', 'fields': ['identifier']},
    'spike':
        {'template': '/Users/katie/likhtik/IG_INED_Safety_Recall/spike/{identifier}', 'fields': ['identifier']},
    'lfp': 
        {'template': '/Users/katie/likhtik/IG_INED_Safety_Recall/{identifier}/{identifier}', 'fields': ['identifier']}  
}

    
exp_info['categorize_neurons'] = {
    'kmeans': True,
    'neuron_types_ascending': ['IN', 'PN'], 
    'neuron_colors': ['blue', 'red'],
    'characteristics': ['fwhm_microseconds', 'firing_rate'],
    'sort_on': 'fwhm_microseconds', 
    'plot': ['fwhm_microseconds', 'firing_rate'],
    'cutoffs': [('fwhm_microseconds', '>', 300, 'PN'), ('fwhm_microseconds', '<=', 300, 'IN')]}


with open(os.path.join(root, 'init_config.json'), 'w', encoding='utf-8') as file:
    json.dump(exp_info, file, ensure_ascii=False, indent=4)