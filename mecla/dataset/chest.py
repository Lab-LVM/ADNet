import json
import os
from glob import glob
from itertools import chain

import numpy as np
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import IMG_EXTENSIONS

# from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from mecla.dataset.factory import register_dataset


@register_dataset
class chexpert(Dataset):
    """
    Reference: https://github.com/Optimization-AI/ICCV2021_DeepAUC
    """
    gray_images = True
    task = 'multilabel'  # choose from 'binary', 'multiclass' or 'multilabel'
    num_labels = 5  # choose None for binary and multiclass

    def __init__(self,
                 root='',
                 mode='train',
                 transform=None,
                 class_index=-1,
                 use_frontal=True,
                 use_upsampling=False,
                 flip_label=False,
                 verbose=False,
                 upsampling_cols=['Cardiomegaly', 'Consolidation'],
                 train_cols=['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'],
                 **kwargs,
                 ):

        # change test to valid b.c. no test split is reserved.
        if mode == 'test':
            mode = 'valid'

        # load data from csv
        self.classes = train_cols
        self.df = pd.read_csv(os.path.join(root, f'{mode}.csv'))
        self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0-small/', '', regex=False)
        self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0/', '', regex=False)
        if use_frontal:
            self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal']

        # upsample selected cols
        if use_upsampling:
            assert isinstance(upsampling_cols, list), 'Input should be list!'
            sampled_df_list = []
            for col in upsampling_cols:
                print('Upsampling %s...' % col)
                sampled_df_list.append(self.df[self.df[col] == 1])
            self.df = pd.concat([self.df] + sampled_df_list, axis=0)

        # impute missing values
        for col in train_cols:
            if col in ['Edema', 'Atelectasis']:
                self.df[col].replace(-1, 1, inplace=True)
                # self.df[col].replace(-1, 0, inplace=True)
                self.df[col].fillna(0, inplace=True)
            elif col in ['Cardiomegaly', 'Consolidation', 'Pleural Effusion']:
                self.df[col].replace(-1, 0, inplace=True)
                self.df[col].fillna(0, inplace=True)
            elif col in ['No Finding', 'Enlarged Cardiomediastinum', 'Lung Opacity', 'Lung Lesion', 'Pneumonia',
                         'Pneumothorax', 'Pleural Other', 'Fracture', 'Support Devices']:  # other labels
                self.df[col].replace(-1, 0, inplace=True)
                self.df[col].fillna(0, inplace=True)
            else:
                self.df[col].fillna(0, inplace=True)

        self._num_images = len(self.df)

        # 0 --> -1
        if flip_label and class_index != -1:  # In multi-class mode we disable this option!
            self.df.replace(0, -1, inplace=True)

        # assert class_index in [-1, 0, 1, 2, 3, 4], 'Out of selection!'
        assert root != '', 'You need to pass the correct location for the dataset!'

        if class_index == -1:  # 5 classes
            if verbose:
                print('Multi-label mode: True, Number of classes: [%d]' % len(train_cols))
                print('-' * 30)
            self.select_cols = train_cols
            self.value_counts_dict = {}
            for class_key, select_col in enumerate(train_cols):
                class_value_counts_dict = self.df[select_col].value_counts().to_dict()
                self.value_counts_dict[class_key] = class_value_counts_dict
        else:
            self.select_cols = [train_cols[class_index]]  # this var determines the number of classes
            self.value_counts_dict = self.df[self.select_cols[0]].value_counts().to_dict()

        self.class_index = class_index
        self.transform = transform

        self._images_list = [root + path for path in self.df['Path'].tolist()]
        if class_index != -1:
            self.targets = self.df[train_cols].values[:, class_index].tolist()
        else:
            self.targets = self.df[train_cols].values.tolist()

        if verbose:
            if class_index != -1:
                if flip_label:
                    self.imratio = self.value_counts_dict[1] / (self.value_counts_dict[-1] + self.value_counts_dict[1])
                    if verbose:
                        print('-' * 30)
                        print('Found %s images in total, %s positive images, %s negative images' % (
                        self._num_images, self.value_counts_dict[1], self.value_counts_dict[-1]))
                        print('%s(C%s): imbalance ratio is %.4f' % (self.select_cols[0], class_index, self.imratio))
                        print('-' * 30)
                else:
                    self.imratio = self.value_counts_dict[1] / (self.value_counts_dict[0] + self.value_counts_dict[1])
                    if verbose:
                        print('-' * 30)
                        print('Found %s images in total, %s positive images, %s negative images' % (
                        self._num_images, self.value_counts_dict[1], self.value_counts_dict[0]))
                        print('%s(C%s): imbalance ratio is %.4f' % (self.select_cols[0], class_index, self.imratio))
                        print('-' * 30)
            else:
                imratio_list = []
                for class_key, select_col in enumerate(train_cols):
                    try:
                        imratio = self.value_counts_dict[class_key][1] / (
                                    self.value_counts_dict[class_key][0] + self.value_counts_dict[class_key][1])
                    except:
                        if len(self.value_counts_dict[class_key]) == 1:
                            only_key = list(self.value_counts_dict[class_key].keys())[0]
                            if only_key == 0:
                                self.value_counts_dict[class_key][1] = 0
                                imratio = 0  # no postive samples
                            else:
                                self.value_counts_dict[class_key][1] = 0
                                imratio = 1  # no negative samples

                    imratio_list.append(imratio)
                    if verbose:
                        # print ('-'*30)
                        print('Found %s images in total, %s positive images, %s negative images' % (
                        self._num_images, self.value_counts_dict[class_key][1], self.value_counts_dict[class_key][0]))
                        print('%s(C%s): imbalance ratio is %.4f' % (select_col, class_key, imratio))
                        print()
                        # print ('-'*30)
                self.imratio = np.mean(imratio_list)
                self.imratio_list = imratio_list

        pos_ratio = np.array(self.targets).mean(axis=0)
        self.weight = np.stack([pos_ratio, 1 - pos_ratio], axis=1)
        self.norm_weight = None

    @property
    def class_counts(self):
        return self.value_counts_dict

    @property
    def imbalance_ratio(self):
        return self.imratio

    @property
    def num_classes(self):
        return len(self.select_cols)

    @property
    def data_size(self):
        return self._num_images

    def __len__(self):
        return self._num_images

    def __getitem__(self, idx):
        image = Image.open(self._images_list[idx])

        if self.transform:
            image = self.transform(image)

        if self.class_index != -1:  # multi-class mode
            label = np.array(self.targets[idx]).reshape(-1).astype(np.float32)
        else:
            label = np.array(self.targets[idx]).reshape(-1).astype(np.float32)

        return image, label


# def split(x, y, ratio):
#     mskf = MultilabelStratifiedKFold(n_splits=int(1 / ratio), shuffle=True, random_state=42)
#     split1, split2 = next(mskf.split(x, y))
#     x1 = x[split1]
#     y1 = y[split1]
#     x2 = x[split2]
#     y2 = y[split2]
#
#     return x1, y1, x2, y2


@register_dataset
class nihchest:
    gray_images = True
    task = 'multilabel'  # choose from 'binary', 'multiclass' or 'multilabel'
    num_labels = 14  # choose None for binary and multiclass

    def __init__(self, root='', mode='train', transform=None):

        self.root = root
        self.transform = transform

        # 1. load csv & set path
        df = pd.read_csv(os.path.join(self.root, 'Data_Entry_2017.csv'))
        img_paths = {os.path.basename(x): x for x in glob(os.path.join(self.root, 'img_512', '*.png'))}
        df['path'] = df['Image Index'].map(img_paths.get)

        # 2. set train flag
        with open(os.path.join(self.root, 'train_val_list.txt'), 'rt') as f:
            train_flag = {x.strip('\n'): 1 for x in f.readlines()}

        with open(os.path.join(self.root, 'test_list.txt'), 'rt') as f:
            train_flag.update({x.strip('\n'): 0 for x in f.readlines()})

        df['train'] = df['Image Index'].map(train_flag.get)

        # 3. change label to one-hot label
        df['Finding Labels'] = df['Finding Labels'].map(lambda x: x.replace('No Finding', ''))
        all_labels = np.unique(list(chain(*df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
        all_labels = [x for x in all_labels if len(x) > 0]

        for c_label in all_labels:
            if len(c_label) > 1:  # leave out empty labels
                df[c_label] = df['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)

        # 4. split dataset into train, test
        df_split = df[df['train'] == (1 if mode == 'train' else 0)]
        self.x = df_split['path'].values.tolist()
        self.y = df_split[all_labels].values
        self.classes = all_labels
        self.norm_weight = np.array(self.y).sum(axis=0) / (np.array(self.y).sum(axis=0) ** 2).sum() ** 0.5
        # self.weight = (1 / np.array(self.y).mean(axis=0)) / (1 / np.array(self.y).sum(axis=0)).mean()
        self.weight = np.stack([1 / (np.array(self.y) == 0).astype(float).sum(axis=0),
                                1 / (np.array(self.y) == 1).astype(float).sum(axis=0)], axis=1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = Image.open(self.x[idx])
        label = np.asarray(self.y[idx]).astype(np.float32)

        if self.transform:
            image = self.transform(image)

        return image, label


def load_json(file_path):
    with open(file_path, 'rt') as f:
        return json.load(f)


@register_dataset
class mimic:
    gray_images = True
    task = 'multilabel'  # choose from 'binary', 'multiclass' or 'multilabel'
    num_labels = 14  # choose None for binary and multiclass
    def __init__(self, root, mode='train', transform=None):
        self.root = root
        self.img_root = os.path.join(root, 'img_384')
        self.transform = transform
        self.classes = ["Atelectasis", "Cardiomegaly", "Consolidation",	"Edema", "Enlarged Cardiomediastinum",
                        "Fracture", "Lung Lesion", "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other",
                        "Pneumonia", "Pneumothorax", "Support Devices"]
        self.weight = None
        self.norm_weight = None

        if mode == 'train':
            self.x = load_json(os.path.join(root, 'train_x.json'))
            self.y = load_json(os.path.join(root, 'train_y.json'))
        else:
            self.x = load_json(os.path.join(root, 'test_x.json'))
            self.y = load_json(os.path.join(root, 'test_y.json'))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        image = Image.open(os.path.join(self.img_root, self.x[item]))
        label = np.asarray(self.y[item]).astype(np.float32)

        if self.transform:
            image = self.transform(image)

        return image, label


@register_dataset
class all:
    gray_images = True
    task = 'multilabel'  # choose from 'binary', 'multiclass' or 'multilabel'
    num_labels = 21  # choose None for binary and multiclass
    def __init__(self, root, mode='train', transform=None):
        self.root = root
        self.img_root = root
        self.transform = transform
        self.classes = [
            'Lung Opacity', 'Atelectasis', 'Cardiomegaly', 'Consolidation',
            'Edema', 'Pleural Effusion', 'Emphysema', 'Enlarged Cardiomediastinum',
            'Fibrosis', 'Fracture', 'Hernia', 'Infiltration',
            'Lung Lesion', 'Mas', 'Nodule', 'No Finding',
            'Pleural Thickening', 'Pleural Other', 'Pneumonia', 'Pneumothorax',
            'Support Devices'
        ]
        self.weight = None
        self.norm_weight = None

        if mode == 'train':
            self.x = load_json(os.path.join(root, 'train_x.json'))
            self.y = load_json(os.path.join(root, 'train_y.json'))
        else:
            self.x = load_json(os.path.join(root, 'test_x.json'))
            self.y = load_json(os.path.join(root, 'test_y.json'))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        image = Image.open(os.path.join(self.root, self.x[item]))
        label = np.asarray(self.y[item]).astype(np.float32)

        if self.transform:
            image = self.transform(image)

        return image, label


@register_dataset
class cxrd(ImageFolder):
    gray_images = True
    task = 'multiclass'
    num_labels = 0

    def __init__(self, root='', mode='train', transform=None, **kwargs):
        super(cxrd, self).__init__(os.path.join(root, mode), transform)


@register_dataset
class covid(ImageFolder):
    gray_images = True
    task = 'multiclass'
    num_labels = 0

    def __init__(self, root='', mode='train', transform=None, **kwargs):
        super(covid, self).__init__(root, transform)
        x_train, x_val, y_train, y_val = train_test_split(self.samples, self.targets,
                                                          test_size=0.2, stratify=self.targets,
                                                          shuffle=True, random_state=2024)
        if mode == 'train':
            self.samples = x_train
            self.targets = y_train
        else:
            self.samples = x_val
            self.targets = y_val

        # self.targets = np.asarray(self.targets).astype(np.float32)


@register_dataset
class polcovid(ImageFolder):
    gray_images = True
    task = 'multiclass'
    num_labels = 0

    def __init__(self, root='', mode='train', transform=None, **kwargs):
        super(polcovid, self).__init__(root, transform)
        x_train, x_val, y_train, y_val = train_test_split(self.samples, self.targets,
                                                          test_size=0.2, stratify=self.targets,
                                                          shuffle=True, random_state=2024)
        if mode == 'train':
            self.samples = x_train
            self.targets = y_train
        else:
            self.samples = x_val
            self.targets = y_val

if __name__ == '__main__':
    ds = mimic(root='../../data/classification/chest/mimic-cxr')
    print(len(ds))
    print(ds.classes)
    print(ds.weight)
