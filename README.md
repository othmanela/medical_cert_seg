# Certified Medical Image Segmentation

This repository contains the implementation of our MICCAI 2023 submission ["Certification of Deep Learning Models for Medical Image Segmentation"](https://arxiv.org/abs/2310.03664).

## Data and models
We evaluated our method on five medical image segmentation datasets. Three lung datasets comprising both single and multi-class segmentation, one dermatology dataset and one polyp dataset.
Follow the instructions below to download the required data:

### Skin Lesion

The dataset is obtained from the [ISIC 2018 challenge](https://challenge2018.isic-archive.com/task1/) and can be 
downloaded [here](https://challenge.isic-archive.com/data#2018).

Download the training and validation input and GT for Task 1 and extract the folders as follows:

```
datasets/
  lesion/
    ISIC2018_Task1-2_Validation_Input/
    ISIC2018_Task1-2_Training_Input/
    ISIC2018_Task1_Validation_GroundTruth/
    ISIC2018_Task1_Training_GroundTruth/
```

Then, navigate to `datasets/lesion` and run `python make_dataset.py`.

### Polyp

The data is obtained from the [CVC-ClinicDB challenge](https://polyp.grand-challenge.org/Databases/).
Processed png images can be found [here](https://www.kaggle.com/balraj98/cvcclinicdb).

Download the dataset and add it as follows:

```
datasets/
  polyp/
    CVC-ClinicDB/
      Original/
        612.png
        ...
      Ground Truth/
        ...
```

Then run `python datasets/polyp/split_dataset.py`.

### Lung
Download the three lung datasets by following the links provided below:
- [JSRT - Japanese Society of Radiological Technology](http://db.jsrt.or.jp/eng.php)
- [Montgomery](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/index.html)
- [Shenzhen](https://www.kaggle.com/datasets/yoctoman/shcxr-lung-mask)

Place them in the project as follows:
```
datasets/
  jsrt/
    train/
      input/
        ...
      label/
        ...
    valid/
      input/
        ...
      label/
        ...
    test/
      input/
        ...
      label/
        ...
  shen/
    ...
  mount/
    ...
```

Follow the same hierarchy for folder `mount` and `shen` which will contain respectively Montgomery and Shenzhen datastets.


### Denoising Diffusion Model
The Denoising Diffusion Probabilistic Models used in the paper is based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion). 
Download the class unconditional pretrained model [here](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt) and place it in the `models` directory.


## Usage

### Setup
First start by installing the requirements of the segmentation by using the `environment.yml`.
If you don't want to use conda, the main requirements for the project are:

- PyTorch 1.7.1
- PyTorch ignite 0.4.3
- segmentation_models_pytorch 0.1.3
- Albumentations 0.5.2
- OpenCV 4.5.1.48


### Usage for multi-class segmentation
#### Launch certification for skin lesions dataset with a Deeplab model
```
python -u test_certify_jsrt.py --weights weights/jsrt_deeplab/best_model.pt --model deeplab --dataset jsrt --multi --sigma 0.25  --denoise
```
#### Launch certification for skin lesions dataset with a Deeplab model and without denoising
```
python -u test_certify_jsrt.py --weights weights/jsrt_deeplab/best_model.pt --model deeplab --dataset jsrt --multi --sigma 0.25
```

### Usage for single-class segmentation
#### Launch certification for skin lesions dataset with a UNet model
```
python -u test_certify.py --weights weights/lesion_unet/best_model.pt --model unet --dataset lesion --sigma 0.25 --denoise
```
#### Launch certification for skin lesions dataset with a UNet model and without denoising
```
python -u test_certify.py --weights weights/lesion_unet/best_model.pt --model unet --dataset lesion --sigma 0.25
```

#### Launch certification for polyp dataset with a ResUNet++ model
```
python -u test_certify.py --weights weights/polyp_resunetpp/best_model.pt --model resunetpp --dataset polyp --sigma 0.25 --denoise
```
#### Launch certification for polyp dataset with a ResUNet++ model and without denoising
```
python -u test_certify.py --weights weights/polyp_resunetpp/best_model.pt --model resunetpp --dataset polyp --sigma 0.25
```

Similar commands can be used for single-class lung segmentation (montgomery and shenzhen datasets).

## Citation
If you find this work useful, please consider citing it:

```
@InProceedings{laousy23miccai,
author="Laousy, Othmane and Araujo, Alexandre and Chassagnon, Guillaume and Paragios, Nikos and Revel, Marie-Pierre and Vakalopoulou, Maria",
editor="Greenspan, Hayit and Madabhushi, Anant and Mousavi, Parvin and Salcudean, Septimiu and Duncan, James and Syeda-Mahmood, Tanveer and Taylor, Russell",
title="Certification of Deep Learning Models for Medical Image Segmentation",
booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2023",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="611--621",
isbn="978-3-031-43901-8"
}

```