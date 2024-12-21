import os
import shutil

# Define the source and destination directories
directory1 = '/Users/katie/likhtik/INED_IG_Safety_Recall'
directory2 = '/Users/katie/likhtik/IG_INED_Safety_Recall'

# Define the extensions to look for
extensions = ('.py', '.npy', '.tsv', '.json')

# List of subdirectories to process
subdirectories = ['IG160', 'IG163', 'IG176', 'IG178', 'IG180', 'IG154', 'IG156', 'IG158', 'IG177', 'IG179']

for subdirectory in subdirectories:
    src_path = os.path.join(directory1, subdirectory)
    dest_path = os.path.join(directory2, subdirectory)

    # Ensure the destination subdirectory exists
    os.makedirs(dest_path, exist_ok=True)

    # Iterate over all files in the source subdirectory
    for filename in os.listdir(src_path):
        # Check if the file has one of the desired extensions
        if filename.endswith(extensions):
            src_file_path = os.path.join(src_path, filename)
            dest_file_path = os.path.join(dest_path, filename)

            # Move the file
            shutil.move(src_file_path, dest_file_path)
            print(f'Moved: {src_file_path} -> {dest_file_path}')
