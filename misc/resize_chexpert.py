import json
import os
from glob import glob
from pathlib import Path

import PIL.Image
from tqdm import tqdm


def main(size=1024):
    save_root = f'/home/hankyul/shared/hdd_ext/hdd3000/CheXpert-v1.0_{size}'
    img_path_list = glob('/home/hankyul/shared/hdd_ext/hdd3000/CheXpert-v1.0/*/*/*/*.jpg')
    shape_list = []

    for img_path in tqdm(img_path_list):
        img = PIL.Image.open(img_path)
        shape_list.append(list(img.size))

        save_path = os.path.join(save_root, *img_path.split('/')[-4:])
        Path(os.path.dirname(save_path)).mkdir(exist_ok=True, parents=True)
        img = img.resize((size, size))
        img.save(save_path)

    print('width:', sum([i[0] for i in shape_list]) / len(shape_list),
          'height:', sum(i[1] for i in shape_list) / len(shape_list))

    with open('chexpert_val_size.json', 'wt') as f:
        json.dump(shape_list, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()