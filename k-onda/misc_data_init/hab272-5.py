
import os
import json

root = '/Users/katie/likhtik/CH27mice'


period_info = {
    'prelight': {'relative': True, 'target': 'light', 'shift': -35, 'duration': 35},
    'light': {'relative': False, 'reference_period_type': 'prelight', 
             'duration': 35, 'code': 65534},
    'instructions': ['periods_from_nev']
}

animals = [
    {'identifier':'CH275', 'period_info': period_info, 'condition': 'foo'}
]

exp_info = {}

exp_info['animals'] = animals
exp_info['conditions'] = ['foo']
exp_info['sampling_rate'] = 30000
exp_info['identifier'] = 'CH27-'
exp_info['path_constructors'] = {
    'nev' : 
        {'template': '/Users/katie/likhtik/CH27mice/{identifier}/{identifier}_HABCTXB.mat', 'fields': ['identifier']},
    'phy': 
        {'template': '/Users/katie/likhtik/CH27mice/{identifier}', 'fields': ['identifier']}   
        }
    
exp_info['get_units_from_phy'] = True


with open(os.path.join(root, 'init_config.json'), 'w', encoding='utf-8') as file:
    json.dump(exp_info, file, ensure_ascii=False, indent=4)


