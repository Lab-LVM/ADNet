import numbers

import PIL.Image
import numpy as np
from torchvision.transforms import functional as F
try:
    import accimage
except ImportError:
    accimage = None
from PIL import Image


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


class MultiCrop:
    def __init__(self, size, n=1, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.n = n

    @staticmethod
    def get_params(img, output_size, n):
        h, w = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i_list = list(np.linspace(0, h - th, n).astype(int))
        j_list = list(np.linspace(0, w - tw, n).astype(int))
        return i_list, j_list, th, tw

    def __call__(self, img):
        if self.padding > 0:
            img = F.pad(img, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))

        i, j, h, w = self.get_params(img, self.size, self.n)

        return n_random_crops(img, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


def n_random_crops(img, xs, ys, h, w):
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    crops = []
    for x in xs:
        for y in ys:
            new_crop = img.crop((y, x, y + w, x + h))
            crops.append(new_crop)
    return tuple(crops)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    img = PIL.Image.open('../../../exp/iu4.jpg')
    img = img.resize((510, 510))
    f = MultiCrop(480, 3)
    crop_imgs = f(img)

    plt.imshow(img)
    plt.show()

    for crop_img in crop_imgs:
        plt.imshow(crop_img)
        plt.show()
