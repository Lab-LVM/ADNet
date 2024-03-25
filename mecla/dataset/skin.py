import os
from glob import glob
from PIL import Image
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split

from mecla.dataset.factory import register_dataset


@register_dataset
class isic2018:
    gray_images = False
    task = 'multiclass'  # choose from 'binary', 'multiclass' or 'multilabel'
    num_labels = None  # choose None for binary and multiclass

    def __init__(self, root='', mode='train', transform=None):

        self.root = os.path.join(root, mode)
        self.transform = transform
        self.classes = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

        if mode in ['train', 'valid']:
            # 1. load csv & set path
            df = pd.read_csv(os.path.join(self.root, 'label.csv'))
            img_paths = {os.path.basename(x).strip('.jpg'): x for x in glob(os.path.join(self.root, 'img', '*.jpg'))}

            # 2. set x, y
            self.x = list(df['image'].map(img_paths.get).values)
            self.y = list(np.argmax(df[self.classes].values, axis=1))

        else:
            self.x = list(glob(os.path.join(self.root, '*.jpg')))
            self.y = [0] * len(self.x)
            self.id_list = [os.path.basename(x).strip('.jpg') for x in glob(os.path.join(self.root, '*.jpg'))]

        self.weight = np.unique(np.array(self.y), return_counts=True)[1]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = Image.open(self.x[idx])
        label = self.y[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


@register_dataset
class isic2019:
    gray_images = False
    task = 'multiclass'  # choose from 'binary', 'multiclass' or 'multilabel'
    num_labels = None  # choose None for binary and multiclass
    norm_weight = None

    def __init__(self, root='', mode='train', transform=None):

        self.root = os.path.join(root, 'train' if mode in ['train', 'valid'] else 'test')
        self.transform = transform
        self.classes = ['MEL','NV','BCC','AK','BKL','DF','VASC','SCC','UNK']

        if mode in ['train', 'valid']:
            # 1. load csv & set path
            df = pd.read_csv(os.path.join(self.root, 'label.csv'))
            img_paths = {os.path.basename(x).strip('.jpg'): x for x in glob(os.path.join(self.root, 'img_512', '*.jpg'))}

            # 2. set x, y
            x = list(df['image'].map(img_paths.get).values)
            y = list(np.argmax(df[self.classes].values, axis=1))

            # 3. split train, val
            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2,
                                                              stratify=y, shuffle=True, random_state=42)
            if mode == 'train':
                self.x = x_train
                self.y = y_train
            else:
                self.x = x_val
                self.y = y_val
        else:
            self.x = list(glob(os.path.join(self.root, '*.jpg')))
            self.y = [0] * len(self.x)
            self.id_list = [os.path.basename(x).strip('.jpg') for x in glob(os.path.join(self.root, '*.jpg'))]

        self.weight = np.unique(np.array(self.y), return_counts=True)[1]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = Image.open(self.x[idx])
        label = self.y[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


@register_dataset
class isic2020:
    gray_images = False
    task = 'binary'  # choose from 'binary', 'multiclass' or 'multilabel'
    num_labels = None  # choose None for binary and multiclass
    norm_weight = None

    def __init__(self, root='', mode='train', transform=None):

        self.root = os.path.join(root, 'train' if mode in ['train', 'valid'] else 'test')
        self.transform = transform
        self.classes = ['benign', 'malignant']

        if mode in ['train', 'valid']:
            # 1. load csv & set path
            df = pd.read_csv(os.path.join(self.root, '../train_metadata.csv'))
            img_paths = {os.path.basename(x).strip('.jpg'): x for x in glob(os.path.join(self.root, '*.jpg'))}

            # 2. set x, y
            x = list(df['image'].map(img_paths.get).values)
            y = list(df['target'].values)

            # 3. split train, val
            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2,
                                                              stratify=y, shuffle=True, random_state=42)
            if mode == 'train':
                self.x = x_train
                self.y = y_train
            else:
                self.x = x_val
                self.y = y_val
        else:
            self.x = list(glob(os.path.join(self.root, '*.jpg')))
            self.y = [0] * len(self.x)
            self.id_list = [os.path.basename(x).strip('.jpg') for x in glob(os.path.join(self.root, '*.jpg'))]

        self.weight = np.unique(np.array(self.y), return_counts=True)[1]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = Image.open(self.x[idx])
        label = self.y[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

if __name__ == '__main__':
    ds = isic2020(root='../../data/classification/skin/ISIC-2020', mode='valid')
    print(ds.weight)