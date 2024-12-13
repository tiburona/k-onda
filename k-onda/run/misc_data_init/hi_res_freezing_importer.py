from utils.utils import load, save
import os
import h5py
import re
from collections import defaultdict


STORE_PATH_ROOT = "/Users/katie/likhtik/IG_INED_Safety_Recall/behavior"

def group_to_dict(group):
    result = {}
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            result[key] = group_to_dict(item)
        elif isinstance(item, h5py.Dataset):
            result[key] = item[()]
        else:
            result[key] = item
    return result


class MyBehaviorImporter:

    def __init__(self, data_file, animal_id):
        self.data_file = data_file
        self.animal_id = animal_id
        self.data_by_animal = defaultdict(dict)

    def import_data(self):
        path = os.path.join(STORE_PATH_ROOT, 'behavior_pkls')
        calc_exists, result, pkl_path = load(path, 'pkl')
        if not calc_exists:
            result = self.get_data_from_mat_file(self.data_file)
        return result

    def get_data_from_mat_file(self):
        data = self.load_mat_file(self.data_file)
        for row in data:
            result = self.parse_row(row)

    def load_mat_file(self, data_file):
         # Load the .mat file
        mat_file = h5py.File(data_file, 'r')
        struct = mat_file['AdHocSideNoMOve'][()]
        return group_to_dict(struct)
    
    def parse_row(self, row):
        animal_id, period_type, period_no = self.parse_name(row['Name'])
        

    
    def parse_name(self, name):
        pattern = r'^(\d{3})([a-zA-Z]+)(\d{3})DLC'
        match = re.match(pattern, name)
        if match:
            an_id, period_type, period_no = match.groups()
            return (f"IG_{an_id}", period_type, int(period_no)-1)

     

