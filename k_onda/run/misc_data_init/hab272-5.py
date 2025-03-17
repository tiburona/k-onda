
import os
import json

root = '/Users/katie/likhtik/CH27mice'

# period_info = {
#     'pretone_plus': {'relative': True, 'target': 'cs_plus', 'shift': -30, 'duration': 30,
#                      'conditions': {'tone': 'off', 'cs': 'plus'}},
#     'pretone_minus': {'relative': True, 'target': 'cs_minus', 'shift': -30, 'duration': 30,
#                       'conditions': {'tone': 'off', 'cs': 'minus'}},
#     'cs_plus': {'relative': False, 'reference_period_type': 'pretone', 'event_duration': 1, 
#              'duration': 30, 'events': {'start': 'period_onset', 'pattern': {'range_args': [30]}},
#              'conditions': {'tone': 'on', 'cs': 'plus'},
#              'nev': {'code': 65503, 'indices': [1, 3, 6, 7, 9, 10]}},
#     'cs_minus': {'relative': False, 'reference_period_type': 'pretone', 'event_duration': 1, 
#              'duration': 30, 'events': {'start': 'period_onset', 'pattern': {'range_args': [30]}},
#              'conditions': {'tone': 'off', 'cs': 'minus'},
#              'nev': {'code': 65503, 'indices': [0, 2, 4, 5, 8, 11]}}
# }

period_info = {
    'prelight': {'relative': True, 'target': 'light', 'shift': -35, 'duration': 35},
    'light': {'relative': False, 'reference_period_type': 'prelight', 
             'duration': 35, 'code': 65534, 'nev': {'code': 65534}}}



units = {'instructions': ['get_units_from_phy']}

animals = [
    {'identifier':'CH275', 'period_info': period_info, 'condition': 'foo', 'units': units},
    {'identifier':'CH272', 'period_info': period_info, 'condition': 'foo', 'units': units}
    
]

exp_info = {}

exp_info['animals'] = animals
exp_info['conditions'] = ['foo']
exp_info['sampling_rate'] = 30000
exp_info['identifier'] = 'CH27mice'
exp_info['path_constructors'] = {
    'nev' : 
        {'template': '/Users/katie/likhtik/CH27mice/{identifier}/{identifier}_HABCTXB.mat', 'fields': ['identifier']},
    'phy': 
        {'template': '/Users/katie/likhtik/CH27mice/{identifier}', 'fields': ['identifier']},
    'spike':
        {'template': '/Users/katie/likhtik/CH27mice/spike/{identifier}', 'fields': ['identifier']}}
    
exp_info['get_units_from_phy'] = True


with open(os.path.join(root, 'init_config.json'), 'w', encoding='utf-8') as file:
    json.dump(exp_info, file, ensure_ascii=False, indent=4)


