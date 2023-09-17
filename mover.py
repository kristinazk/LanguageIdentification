import os
import shutil
import random

source_folder = 'PATH_TO_SOURCE_FOLDER'
destination_folder = 'PATH_TO_DEST_FOLDER'

all_files = os.listdir(source_folder)

jpg_files = [file for file in all_files if file.endswith('jpg')]

n = 16

random_numbers = []

extensions = ['jpg', 'xml']

appeared_names = []

for _ in range(n):
    name = random.choice(jpg_files)[:-4]

    if name not in appeared_names:
        for ext in extensions:
            shutil.move(os.path.join(source_folder, f'{name}.{ext}'), destination_folder)

    appeared_names.append(name)
