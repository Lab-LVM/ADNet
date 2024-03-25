import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

from mecla.dataset.chest import chexpert, nihchest, mimic

# 1. create root folder, symbolic link
root = "/home/hankyul/shared/medical/classification/chest/all"
new_cxp_path = os.path.join(root, 'cxp')
new_nih_path = os.path.join(root, 'nih')
new_mimic_path = os.path.join(root, 'mimic')
Path(root).mkdir(exist_ok=True)

if not os.path.exists(new_cxp_path):
    os.symlink("/home/hankyul/shared/medical/classification/chest/cheXpert/CheXpert-v1.0-small", new_cxp_path)
if not os.path.exists(new_nih_path):
    os.symlink("/home/hankyul/shared/medical/classification/chest/nih", new_nih_path)
if not os.path.exists(new_mimic_path):
    os.symlink("/home/hankyul/shared/medical/classification/chest/mimic-cxr", new_mimic_path)

# 2. create label_convertor
total_class_names = [
    'Lung Opacity', 'Atelectasis', 'Cardiomegaly', 'Consolidation',
    'Edema', 'Pleural Effusion', 'Emphysema', 'Enlarged Cardiomediastinum',
    'Fibrosis', 'Fracture', 'Hernia', 'Infiltration',
    'Lung Lesion', 'Mas', 'Nodule', 'No Finding',
    'Pleural Thickening', 'Pleural Other', 'Pneumonia', 'Pneumothorax',
    'Support Devices'
]

chexpert_class_names = [
    'Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis',
    'Pleural Effusion', 'No Finding', 'Enlarged Cardiomediastinum', 'Lung Opacity',
    'Lung Lesion', 'Pneumonia', 'Pneumothorax', 'Pleural Other',
    'Fracture', 'Support Devices',
]

mimic_class_names = [
    "Atelectasis", "Cardiomegaly", "Consolidation",	"Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
    "Pneumothorax", "Support Devices"
]

nih_class_names = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Pleural Effusion', 'Emphysema', 'Fibrosis', 'Hernia',
    'Infiltration', 'Mas', 'Nodule', 'Pleural Thickening',
    'Pneumonia', 'Pneumothorax'
]

cxp2all = [total_class_names.index(class_name) for class_name in chexpert_class_names]
mimic2all = [total_class_names.index(class_name) for class_name in mimic_class_names]
nih2all = [total_class_names.index(class_name) for class_name in nih_class_names]

def label_converter(old_label, mapping_fn):
    new_label = [0 for _ in range(len(total_class_names))]
    not_found_flag = False
    for idx, val in enumerate(old_label):
        if val:
            not_found_flag = True
            new_label[mapping_fn[idx]] = 1

    if not_found_flag: # for NIH dataset
        new_label[-6] = 1

    return new_label

def collect_dataset(new_x, new_y, old_x, old_y, mapping_fn, old_root, new_root):
    for i in tqdm(range(len(old_x))):
        new_x.append(os.path.join(new_root, old_x[i].replace(old_root, '')))
        new_y.append(label_converter(old_y[i], mapping_fn))

def save_as_json(data, file_name):
    file_name = os.path.join(root, file_name)
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


train_x = []
train_y = []
val_x = []
val_y = []

# 3. append chexpert, mimic, nih to train/val list
train_dataset = chexpert("../data/classification/chest/cheXpert/CheXpert-v1.0-small/", mode='train', train_cols=chexpert_class_names)
val_dataset = chexpert("../data/classification/chest/cheXpert/CheXpert-v1.0-small/", mode='valid', train_cols=chexpert_class_names)
collect_dataset(train_x, train_y, train_dataset._images_list, train_dataset.targets, cxp2all,
                '../data/classification/chest/cheXpert/CheXpert-v1.0-small/', 'cxp/')
collect_dataset(val_x, val_y, val_dataset._images_list, val_dataset.targets, cxp2all,
                '../data/classification/chest/cheXpert/CheXpert-v1.0-small/', 'cxp/')

train_dataset = mimic("../data/classification/chest/mimic-cxr/", mode='train')
val_dataset = mimic("../data/classification/chest/mimic-cxr/", mode='valid')
collect_dataset(train_x, train_y, train_dataset.x, train_dataset.y, mimic2all, '', 'mimic/img_384')
collect_dataset(val_x, val_y, val_dataset.x, val_dataset.y, mimic2all, '', 'mimic/img_384')

train_dataset = nihchest("../data/classification/chest/nih/", mode='train')
val_dataset = nihchest("../data/classification/chest/nih/", mode='val')
collect_dataset(train_x, train_y, train_dataset.x, train_dataset.y, nih2all, '../data/classification/chest/nih/', 'nih')
collect_dataset(val_x, val_y, val_dataset.x, val_dataset.y, nih2all, '../data/classification/chest/nih/', 'nih')

save_as_json(train_x, 'train_x.json')
save_as_json(train_y, 'train_y.json')
save_as_json(val_x, 'valid_x.json')
save_as_json(val_y, 'valid_y.json')
