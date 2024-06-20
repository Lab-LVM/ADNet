# MIMIC-CXR dataset preparation

This document contains MIMIC-CXR dataset preparation guidelines.

1. Download MIMIC-CXR-JPG dataset from [PhysioNet](https://physionet.org/content/mimic-cxr/2.0.0/).
2. Resize input image resolution to 384x384 size using below python script. This script resize and save images to `data_root/img_384` folder. In the provided script, you should change `data_root` to your local mimic-cxr dataset root path.
We assume images are under `files` directory of your mimic-cxr dataset folder like this:
   ```bash
    hankyul@server1:~/mimic-cxr/files$ tree -L 3 | head -n 20
    .
    ├── index.html
    ├── p10
    ├── p10000032
    │   ├── index.html
    │   ├── s50414267
    │   │   ├── 02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg
    │   │   ├── 174413ec-4ec4c1f7-34ea26b7-c5f994f8-79ef1962.jpg
    │   │   └── index.html
    │   ├── s53189527
    │   │   ├── 2a2277a9-b0ded155-c0de8eb9-c124d10e-82c5caab.jpg
    │   │   ├── e084de3b-be89b11e-20fe3f9f-9c8d8dfe-4cfd202c.jpg
    │   │   └── index.html
    │   ├── s53911762
    │   │   ├── 68b5c4b1-227d0485-9cc38c3f-7b84ab51-4b472714.jpg
    │   │   ├── fffabebf-74fd3a1f-673b6b41-96ec0ac9-2ab69818.jpg
    │   │   └── index.html
    │   └── s56699142
    │       ├── ea030e7a-2e3b1346-bc518786-7a8fd698-f673b44c.jpg
    │       └── index.html
    ```

    ```python
    import json
    import os
    from glob import glob
    from pathlib import Path
    
    import PIL.Image
    from tqdm import tqdm
    
    
    def main(size=384):
        data_root = '/path/to/mimix-cxr'
        save_root = os.path.join(data_root, 'img_384')
        img_path_list = glob(os.paht.join(data_root, 'files/*/*/*.jpg'))
        shape_list = []
    
        for img_path in tqdm(img_path_list):
            img = PIL.Image.open(img_path)
            shape_list.append(list(img.size))
    
            save_path = os.path.join(save_root, *img_path.split('/')[-3:])
            Path(os.path.dirname(save_path)).mkdir(exist_ok=True, parents=True)
            img = img.resize((size, size))
            img.save(save_path)
    
        print('width:', sum([i[0] for i in shape_list]) / len(shape_list),
              'height:', sum(i[1] for i in shape_list) / len(shape_list))
    
        with open('mimic_size.json', 'wt') as f:
            json.dump(shape_list, f, ensure_ascii=False, indent=4)
    
    if __name__ == '__main__':
        main()
    ```
    
3. Make image filename list for train and test split (`train_x.json`, `test_x.json`). Whole images under `img_384` are split 8:2 ratio, ensuring no patients overlap occurs.
We use patient id list from [CheXclusion](https://github.com/LalehSeyyed/CheXclusion/blob/main/MIMIC/testSet_SubjID.csv).

    ```json
    [
        "p14839955/s50259034/09785b6c-723e6127-5fd548a2-38f16c83-317f7809.jpg",
        "p14839955/s50259034/313962b1-eefbd747-e4558b29-e26b2c55-593d53fe.jpg",
        "p14839955/s52687617/24060e0e-f0e96fb0-e137e0d0-74c32049-a10805db.jpg",
        "p14839955/s52687617/c31a03e9-8764837f-95306ebd-b64a701c-05d4f308.jpg",
        "p11087914/s50795246/0590a951-63f8dcff-04d2957a-ac74baf1-ea81e311.jpg",
    ]
    ```

4. Make label list for train and test split each (`train_y.json`, `test_y.json`). Each label is one-hot encoded like below.

    ```json
    [
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0
        ]
    ]
    ```
