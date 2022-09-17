import matplotlib.pyplot as plt
import json
import numpy as np


def show_img(img):
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.show()


def save_data(data, name='data'):
    if not isinstance(data, list):
        data = data.tolist()
    with open(f'{name}.json', 'w') as out_file:
        json.dump({name: data}, out_file)
    print(f'file: "{name}.json" was saved')


def load_data(fp, key):
    with open('saved_g.json', 'r') as out_file:
        data = np.array(json.load(out_file)[key])
    return data
