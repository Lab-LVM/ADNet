# ADNet

This repository contains PyTorch-based implementation for "Attentional Decoder Networks for Chest X-ray Image Recognition on High-resolution Features." Our ADNet aims to enhance chest X-ray performance via a lightweight attentional decoder network and harmonic magnitude transforms. We provide model, training, and evaluation (supervised, `k`-NN classifier) code as well as pre-trained weight in the hope that researchers and practitioners widely use our works effortlessly.

<p align="center">
    <img width="850px" src="./misc/method_overview1.png"/>
    <br/>
  <h4 align="center">Method Overview of Attentionl Decoder Network</h4>
</p>

## Tutorial

### 1️⃣ [Inference Only]

1. Copy [mecla/model/convnext.py](mecla/model/convnext.py) into your project root folder.

2. Install `timm==0.6.2` and run the following code snippets.

   ```python
   !wget https://raw.githubusercontent.com/Lab-LVM/ADNet/main/mecla/model/convnext.py
   !wget https://raw.githubusercontent.com/Lab-LVM/ADNet/main/misc/sample.jpg
   import convnext
   from timm import create_model
   from PIL import Image
   from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor
   
   img = Image('sample.jpg')
   x = Compose([
       Resize(438, 3), CenterCrop(416), ToTensor(), 
       Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
   ])(img)
   model = create_model('ResNet50_ADNet_ALL_NIH', pretrained=True)
   y = model(x.unsqueeze(0))
   print(y)
   # output: (...)
   ```
   
   

### 2️⃣ [Train]

1. Clone this repository and install the dependency.

   ```bash
   git clone https://github.com/Lab-LVM/ADNet
   pip install -r requirements.txt
   ```
   
2. Download chest X-ray datasets and change `data_dir` in `config/train.json` .

   ```json
   "data_dir": {
     "chexpert": "/path/to/CheXpert-v1.0-small/",
     "nihchest": "/path/to/nih/",
     "mimic": "/path/to/mimic-cxr/",
   }
   ```

   You can download each dataset using the provided URLs, and the dataset directory looks like the below:

   *Tip.*

   - We resize images to 384x384 resolution using [misc/resize_chexpert.py](misc/resize_chexpert.py).
   - We split MIMIC-CXR using [CheXclusion](https://github.com/LalehSeyyed/CheXclusion) protocols. You can download the test patient ID lists from [here](https://github.com/LalehSeyyed/CheXclusion/blob/main/MIMIC/testSet_SubjID.csv).

   ```bash
   # 1. CheXpert: https://stanfordmlgroup.github.io/competitions/chexpert/
   CheXpert-v1.0-small
   ├── train
   ├── train.csv
   ├── valid
   └── valid.csv
   
   # 2. MIMIC-CXR: https://physionet.org/content/mimic-cxr/2.0.0/
   mimic-cxr
   ├── files
   ├── img_384
   ├── test_x.json
   ├── test_y.json
   ├── train_x.json
   └── train_y.json
   
   # 3. https://www.kaggle.com/datasets/nih-chest-xrays/data
   nih
   ├── Data_Entry_2017.csv
   ├── img_384
   ├── test_list.txt
   └── train_val_list.txt
   ```

3. Run the following command.

   - Download a pre-trained checkpoint and modify the checkpoint path of `model_weight` in [config/train.json](config/train.json). You can choose a checkpoint depending on pre-trained dataset types: ImageNet or NIH+MIMIC+CheXpert.
   
     | No   | Backbone   | ImageNet | NIH+MIMIC+CheXpert |
     | ---- | ---------- | -------- | ------------------ |
     | 1    | ResNet50   |          |                    |
     | 2    | ConvNeXt-T |          |                    |
   
   - See more training scripts in [script/supervised/](script/supervised).
   
   - See available training settings in [config/train.json](config/train.json).
   
   - Add `--use-wandb` for watching progress in [wandb](https://wandb.ai/).
   
   ```bash
   # pattern: python3 train.py -c [GPU Device] -s [Setting Name] -m [Model Name]
   python3 train.py -s nihchest_v13 -c 1 -m convnext_tiny_up2_attn_hmp_crop15_aug
   ```
   
   



## Experiment Result

We provide experiment results with pre-trained checkpoints.

| Train Data         | Test Data | Model            | Image | AUROC | F1    | Recall | Download |
| ------------------ | --------- | ---------------- | ----- | ----- | ----- | ------ | -------- |
| NIH                | NIH       | ResNet50+ADNet   | 448   | 83.49 | 15.70 | 10.77  |          |
| NIH                | NIH       | ConvNeXt-T+ADNet | 448   | 83.80 | 24.23 | 18.46  |          |
| MIMIC              | MIMIC     | ResNet50+ADNet   | 256   | 84.48 | 31.85 | 26.84  |          |
| MIMIC              | MIMIC     | ConvNeXt-T+ADNet | 416   | 85.31 | 31.37 | 26.97  |          |
| CheXpert           | CheXpert  | ResNet50+ADNet   | 256   | 90.60 | -     | -      |          |
| NIH+MIMIC+CheXpert | NIH       | ResNet50+ADNet   | 416   |       |       |        |          |
| NIH+MIMIC+CheXpert | NIH       | ConvNeXt-T+ADNet | 416   |       |       |        |          |
| NIH+MIMIC+CheXpert | MIMIC     | ResNet50+ADNet   | 416   |       |       |        |          |
| NIH+MIMIC+CheXpert | MIMIC     | ConvNeXt-T+ADNet | 416   |       |       |        |          |
| NIH+MIMIC+CheXpert | CheXpert  | ResNet50+ADNet   | 416   |       |       |        |          |

*Note*

- We run our experiments on `Ubuntu 18.04 LTS`, `CUDA 11.03`, and a single NVIDIA RTX 3090 24GB GPU.



## Acknowledgment

Our code is based on the [TIMM library](https://github.com/huggingface/pytorch-image-models).
