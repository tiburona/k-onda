import os
from shutil import copyfile

src_dir = '/Users/katie/likhtik/IG_INED_SAFETY_RECALL/'
dst_dir = '/Users/katie/likhtik/likhtik_scripts/spike_data_processing/documentation/tutorials/psth/data'

animals = ['IG177', 'IG178', 'IG179', 'IG180']

fname = 'channel_positions.npy'


for animal in animals:
    src = os.path.join(src_dir, animal, fname)
    dst = os.path.join(dst_dir, animal, fname)
    copyfile(src, dst)