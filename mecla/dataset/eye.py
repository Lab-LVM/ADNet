import os
from glob import glob
from PIL import Image
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split

from .factory import register_dataset


@register_dataset
class eyepacs:
    gray_images = False
    task = 'multiclass'  # choose from 'binary', 'multiclass' or 'multilabel'
    num_labels = None  # choose None for binary and multiclass

    def __init__(self, root='', mode='train', transform=None):

        self.root = os.path.join(root, 'train')
        self.transform = transform

        # 1. load csv & set path
        df = pd.read_csv(os.path.join(self.root, 'label.csv'))
        img_paths = {os.path.basename(x).strip('.jpeg'): x for x in glob(os.path.join(self.root, 'img_512', '*.jpeg'))}

        # 2. set x, y
        x = list(df['image'].map(img_paths.get).values)
        y = list(df['level'].values)

        # 3. split train, val
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, stratify=y,
                                                          shuffle=True, random_state=42)
        if mode == 'train':
            self.x = x_train
            self.y = y_train
        else:
            self.x = x_val
            self.y = y_val

        self.classes = list(set(self.y))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = Image.open(self.x[idx])
        label = self.y[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


@register_dataset
class messidor2:
    gray_images = False
    task = 'multiclass'  # choose from 'binary', 'multiclass' or 'multilabel'
    num_labels = None  # choose None for binary and multiclass

    def __init__(self, root='', mode='train', transform=None):

        self.root = root
        self.transform = transform

        # 1. load csv & set path
        df = pd.read_csv(os.path.join(self.root, 'label.csv'))
        img_paths = {os.path.basename(x): x for x in glob(os.path.join(self.root, 'img', '*.*'))}

        # 2. set x, y
        x = list(df['id_code'].map(img_paths.get).values)
        y = list(df['diagnosis'].values)

        # 3. split train, val
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, stratify=y,
                                                          shuffle=True, random_state=42)
        if mode == 'train':
            self.x = x_train
            self.y = y_train
        else:
            self.x = x_val
            self.y = y_val

        self.classes = list(set(self.y))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = Image.open(self.x[idx])
        label = self.y[idx]

        if self.transform:
            image = self.transform(image)

        return image, label