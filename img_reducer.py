from PIL import Image
import os
import cv2

directory_path = 'PATH_TO_DIRECTORY'

all_files = os.listdir(directory_path)
xml_files = [file for file in all_files if file.endswith('.xml')]

jpg_files = [file for file in all_files if file.endswith('.jpg')]

print(len(xml_files), len(jpg_files))

for file in jpg_files:
    full_path = os.path.join(directory_path, file)

    image = Image.open(full_path)

    width, height = image.size

    image.resize((width, height), Image.Resampling.HAMMING).save(full_path)

    cv2.imread(full_path)

print(len(jpg_files))




