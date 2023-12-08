import os
import argparse
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folder') # either upper or lower

args = parser.parse_args()

folder = args.folder

for file in os.listdir(folder):
    print('file name: ', file)
    if not file.endswith('.png'):
        name = file.split('.')[0]
        image = Image.open(os.path.join(folder, file))
        image.save(os.path.join(folder, f'{name}.png'))
        os.remove(os.path.join(folder, file))
