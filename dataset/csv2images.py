# Be carefull while running this file
# This file will overwrite existing images

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from os.path import join

data = pd.read_csv('icml_face_data.csv')
data.columns = ['emotion', 'usage', 'pixels']

def prepare_data(data):
    """ Prepare data for modeling 
        input: data frame with labels und pixel data
        output: image and label array """
    
    image_array = np.zeros(shape=(len(data), 48, 48))
    image_label = np.array(list(map(int, data['emotion'])))
    
    for i, row in enumerate(data.index):
        image = np.fromstring(data.loc[row, 'pixels'], dtype=int, sep=' ')
        image = np.reshape(image, (48, 48))
        image_array[i] = image
        
    return image_array, image_label

emotions = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}

train_data = data[data['usage']=='Training']
val_data = data[data['usage']=='PrivateTest']
test_data = data[data['usage']=='PublicTest']


train_img, train_img_label = prepare_data(train_data)
val_img, val_img_label = prepare_data(val_data)
test_img, test_img_label = prepare_data(test_data)

j0,j1,j2,j3,j4,j5,j6 = 0,0,0,0,0,0,0
for i, j in zip(train_img, train_img_label):
    if j == 0:
        path = f'images/train/{emotions[j]}'
        name = f'{emotions[j]}{j0}.png'
        full_path = join(path, name)
        print(full_path)
        plt.imsave(full_path, i, cmap='gray')
        j0 += 1

    if j == 1:
        path = f'images/train/{emotions[j]}'
        name = f'{emotions[j]}{j1}.png'
        full_path = join(path, name)
        print(full_path)
        plt.imsave(full_path, i, cmap='gray')
        j1 += 1

    if j == 2:
        path = f'images/train/{emotions[j]}'
        name = f'{emotions[j]}{j2}.png'
        full_path = join(path, name)
        print(full_path)
        plt.imsave(full_path, i, cmap='gray')
        j2 += 1

    if j == 3:
        path = f'images/train/{emotions[j]}'
        name = f'{emotions[j]}{j3}.png'
        full_path = join(path, name)
        print(full_path)
        plt.imsave(full_path, i, cmap='gray')
        j3 += 1

    if j == 4:
        path = f'images/train/{emotions[j]}'
        name = f'{emotions[j]}{j4}.png'
        full_path = join(path, name)
        print(full_path)
        plt.imsave(full_path, i, cmap='gray')
        j4 += 1

    if j == 5:
        path = f'images/train/{emotions[j]}'
        name = f'{emotions[j]}{j5}.png'
        full_path = join(path, name)
        print(full_path)
        plt.imsave(full_path, i, cmap='gray')
        j5 += 1

    if j == 6:
        path = f'images/train/{emotions[j]}'
        name = f'{emotions[j]}{j6}.png'
        full_path = join(path, name)
        print(full_path)
        plt.imsave(full_path, i, cmap='gray')
        j6 += 1


j0,j1,j2,j3,j4,j5,j6 = 0,0,0,0,0,0,0
for i, j in zip(test_img, test_img_label):
    if j == 0:
        path = f'images/test/{emotions[j]}'
        name = f'{emotions[j]}{j0}.png'
        full_path = join(path, name)
        print(full_path)
        plt.imsave(full_path, i, cmap='gray')
        j0 += 1

    if j == 1:
        path = f'images/test/{emotions[j]}'
        name = f'{emotions[j]}{j1}.png'
        full_path = join(path, name)
        print(full_path)
        plt.imsave(full_path, i, cmap='gray')
        j1 += 1

    if j == 2:
        path = f'images/test/{emotions[j]}'
        name = f'{emotions[j]}{j2}.png'
        full_path = join(path, name)
        print(full_path)
        plt.imsave(full_path, i, cmap='gray')
        j2 += 1

    if j == 3:
        path = f'images/test/{emotions[j]}'
        name = f'{emotions[j]}{j3}.png'
        full_path = join(path, name)
        print(full_path)
        plt.imsave(full_path, i, cmap='gray')
        j3 += 1

    if j == 4:
        path = f'images/test/{emotions[j]}'
        name = f'{emotions[j]}{j4}.png'
        full_path = join(path, name)
        print(full_path)
        plt.imsave(full_path, i, cmap='gray')
        j4 += 1

    if j == 5:
        path = f'images/test/{emotions[j]}'
        name = f'{emotions[j]}{j5}.png'
        full_path = join(path, name)
        print(full_path)
        plt.imsave(full_path, i, cmap='gray')
        j5 += 1

    if j == 6:
        path = f'images/test/{emotions[j]}'
        name = f'{emotions[j]}{j6}.png'
        full_path = join(path, name)
        print(full_path)
        plt.imsave(full_path, i, cmap='gray')
        j6 += 1


j0,j1,j2,j3,j4,j5,j6 = 0,0,0,0,0,0,0
for i, j in zip(val_img, val_img_label):
    if j == 0:
        path = f'images/val/{emotions[j]}'
        name = f'{emotions[j]}{j0}.png'
        full_path = join(path, name)
        print(full_path)
        plt.imsave(full_path, i, cmap='gray')
        j0 += 1

    if j == 1:
        path = f'images/val/{emotions[j]}'
        name = f'{emotions[j]}{j1}.png'
        full_path = join(path, name)
        print(full_path)
        plt.imsave(full_path, i, cmap='gray')
        j1 += 1

    if j == 2:
        path = f'images/val/{emotions[j]}'
        name = f'{emotions[j]}{j2}.png'
        full_path = join(path, name)
        print(full_path)
        plt.imsave(full_path, i, cmap='gray')
        j2 += 1

    if j == 3:
        path = f'images/val/{emotions[j]}'
        name = f'{emotions[j]}{j3}.png'
        full_path = join(path, name)
        print(full_path)
        plt.imsave(full_path, i, cmap='gray')
        j3 += 1

    if j == 4:
        path = f'images/val/{emotions[j]}'
        name = f'{emotions[j]}{j4}.png'
        full_path = join(path, name)
        print(full_path)
        plt.imsave(full_path, i, cmap='gray')
        j4 += 1

    if j == 5:
        path = f'images/val/{emotions[j]}'
        name = f'{emotions[j]}{j5}.png'
        full_path = join(path, name)
        print(full_path)
        plt.imsave(full_path, i, cmap='gray')
        j5 += 1

    if j == 6:
        path = f'images/val/{emotions[j]}'
        name = f'{emotions[j]}{j6}.png'
        full_path = join(path, name)
        print(full_path)
        plt.imsave(full_path, i, cmap='gray')
        j6 += 1