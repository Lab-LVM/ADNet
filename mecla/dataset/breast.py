import json
import os.path

import pandas as pd
from torchvision.datasets import ImageFolder

from mecla.dataset.factory import register_dataset


@register_dataset
class ddsm(ImageFolder):
    gray_images = True
    task = 'binary'
    num_labels = None

    def __init__(self, root='', mode='train', transform=None, **kwargs):
        super(ddsm, self).__init__(os.path.join(root, mode), transform)


@register_dataset
class vindr:
    def __init__(self):
        pass

class cbis_ddsm(ImageFolder):
    gray_images = True
    task = 'binary'
    num_labels = None

    def __init__(self, root='', mode='train', transform=None, **kwargs):
        super(cbis_ddsm, self).__init__(root, transform)
        info = json.load(os.path.join(root, 'info.json'))
        x, label, pid = [[x[i] for x in info] for i in range(3)]
        # self.samples
        # self.targets




if __name__ == '__main__':
    ds = cbis_ddsm('../../data/classification/breast/CBIS-DDSM', mode='train')