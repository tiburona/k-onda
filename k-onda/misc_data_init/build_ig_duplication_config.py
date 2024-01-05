import json
import os

root = '/Users/katie/likhtik/IG_INED_SAFETY_RECALL'

tone_on_code = 65502

ined_lfp_electrodes = {'bla': [3], 'bf': [12], 'pl': [15]}

control_ined = ['INED18', 'INED17', 'INED16', 'INED05', 'INED04']
defeat_ined = ['INED06', 'INED07', 'INED09', 'INED11', 'INED12']

control_ined_dict = {animal: {'lfp_electrodes': ined_lfp_electrodes, 'condition': 'control'} for animal in control_ined}
defeat_ined_dict = {animal: {'lfp_electrodes': ined_lfp_electrodes, 'condition': 'defeat'} for animal in control_ined}
animals = {**control_ined_dict, **defeat_ined_dict}

pl_electrodes = {
    'IG154': (4, 6), 'IG155': (12, 14), 'IG156': (12, 14), 'IG158': (7, 14), 'IG160': (1, 8), 'IG161': (9, 11),
    'IG162': (13, 3), 'IG163': (14, 8), 'IG175': (15, 4), 'IG176': (11, 12), 'IG177': (15, 4), 'IG178': (6, 14),
    'IG179': (13, 15), 'IG180': (15, 4)
}

defeat_ig = ['IG154', 'IG155', 'IG156', 'IG158', 'IG172', 'IG174', 'IG175', 'IG179']
control_ig = ['IG154', 'IG155', 'IG156', 'IG158', 'IG172', 'IG174', 'IG175', 'IG179']

pl_electrodes = {
    'IG154': (4, 6), 'IG155': (12, 14), 'IG156': (12, 14), 'IG158': (7, 14), 'IG160': (1, 8), 'IG161': (9, 11),
    'IG162': (13, 3), 'IG163': (14, 8), 'IG175': (15, 4), 'IG176': (11, 12), 'IG177': (15, 4), 'IG178': (6, 14),
    'IG179': (13, 15), 'IG180': (15, 4)
}

# animals = spike_electrode_animals.keys() + pl_electrodes.keys()

full_data = {
    'conditions': ['defeat', 'control'],
    'identifier': 'IG_INED_Safety_Recall_Duplication',
    'neuron_types': [],
    'lfp_sampling_rate': 200,
    'lfp_root': root,
    'lfp_path_constructor': [],
    'lfp_electrodes': {'hpc': 0, 'bla': 2},
    'lfp_from_stereotrodes': {'nsx_num': 6, 'pl': {}},
    'lost_signal': .75

}

for animal in animals:
    animal_file = os.path.join(root. animal + '.json')
    with open(animal_file, 'r', encoding='utf-8') as file:
        data = file.read()
        json_data = json.loads(data)
        time_stamps = json_data['NEV']['Data']['SerialDigitalIO']['TimeStamp']
        unparsed_data = json_data['NEV']['Data']['SerialDigitalIO']['UnparsedData']
        tone_onsets = [ts for i, ts in enumerate(time_stamps) if unparsed_data[i] == tone_on_code]
        events = [[onset + i * 30000 for i in range(30)] for onset in tone_onsets]
        pretone = {'reference': True, 'target': 'tone', 'shift': 30, 'duration': 30}






