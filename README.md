# Med-USIS: Unsupervised Semantic Image Synthesis for Medical Imaging


Obtaining large labeled datasets in the medical field is often hard due to privacy concerns. Our
approach leverages a unique dataset comprising labeled CT scans with corresponding semantic labels and an unlabeled MR dataset without semantic annotations.
The primary goal is to facilitate the translation of 2D CT semantic maps to 2D MR images.  By stripping away directly identifiable personal
information and focusing on medically relevant, de-identified data (semantic maps), this approach not only
complies with legal and ethical standards but also minimizes the privacy risks associated
with data exposure during sharing or publicationwe introduce an innovative unsupervised method uti-
lizing unpaired images to train our GAN model, termed Med-USIS, based on several datasets. Through quantitative evaluations, Med-USIS has proven its efficacy
in synthesizing MR images that closely approximate actual MR scans in terms of quality Several 
![img.png](https://github.com/TWWinde/Medical-Images-Synthesis/blob/main/assert/ctvsmri.png)

## Med-USIS Framework

This repository is about part of my master research project (Forschungsarbeit), which aims at generating realistic looking medical images from semantic label maps. 
In addition, many different images can be generated from any given label map by simply resampling a noise vector.
We implemented Wavelet-generator, which is based on SPADE and Wavelet-discriminator. 
This repo is unsupervised medical image synthesis, using CT labels to generate MR images. The architecture is shown below.

![img.png](https://github.com/TWWinde/Med-USIS/blob/main/images/model.png)

## Parameter Count of Model Components

| Name    | OASIS Generator | Wavelet Generator | Wavelet Discriminator | U-Net  |
|---------|-----------------|-------------------|-----------------------|--------|
| #param  | 71.1M           | 55.8M             | 26.2M                 | 22.3M  |
## Setup
First, clone this repository:
```
git clone https://github.com/TWWinde/Med-USIS.git
cd Med-USIS
```

The basic requirements are PyTorch and Torchvision.
```
conda env create Med-USIS
source activate Med-USIS
```
## Datasets

We implement our models based on [AutoPET](https://autopet.grand-challenge.org), which is used for paired supervised model(this repo), and [SynthRAD2023](https://synthrad2023.grand-challenge.org), which is used for unpaired unsupervised model.

## Input Pipeline
For medical images, the pre-processing is of great importance.
Execute ```dataloaders/generate_2d_images.py```to transfer 3d niffti images to slices(2d labels and RGB images).
implementing ```remove_background```function can remove the useless artifacts from medical equipment 
![img.png](https://github.com/TWWinde/Med-USIS/blob/main/images/removebackground.png)
The script above results in the following folder structure.

```
data_dir
├── train
|     ├──images
|     └── labels                 
├── test
|     ├──images 
|     └── labels
└── val
      ├──images
      └── labels
```

## Training the model

To train the model, execute the training scripts through ```sbatch batch.sh``` . 
In these scripts you first need to specify the path to the data folder. 
Via the ```--name``` parameter the experiment can be given a unique identifier. 
The experimental results are then saved in the folder ```./checkpoints```, where a new folder for each run is created with the specified experiment name. 
You can also specify another folder for the checkpoints using the ```--checkpoints_dir``` parameter.
If you want to continue training, start the respective script with the ```--continue_train``` flag. 
Have a look at ```config.py``` for other options you can specify.  
Training on 1 NVIDIA A5000 (32GB) is recommended.


## Testing the model

To test a trained model, execute the testing scripts in the ```scripts``` folder. The ```--name``` parameter 
should correspond to the experiment name that you want to test, and the ```--checkpoints_dir``` should the folder 
where the experiment is saved (default: ```./checkpoints```). These scripts will generate images from a pretrained model 
in ```./results/name/```.


## Measuring Metrics

The FID, PIPS, PSNR, RMSE and SSIM are computed on the fly during training, using the popular PyTorch implementation from https://github.com/mseitzer/pytorch-fid. 
At the beginning of training, the inception moments of the real images are computed before the actual training loop starts. 
How frequently the FID should be evaluated is controlled via the parameter ```--freq_fid```, which is set to 5000 steps by default.
The inception net that is used for FID computation automatically downloads a pre-trained inception net checkpoint. 
The VggNet that is used for LPIPs computation automatically downloads a pre-trained VggNet checkpoint. 

In oder to compute MIoU, we use the powerful segmentation benchmark--nnUnet. We trained on AutoPET 2d slices and our validation Dice reached 0.78.
The checkpoints for the pre-trained segmentation model is available [here](). For the major classed, the MIoU are more the 0.7. The code of nnUnet id loacted
in my another [repo](https://github.com/TWWinde/nnUNet). After configuration, you can simply execute ```utils/miou_folder/nnunet_segment.py```
to compute the MIoU.

## Pretrained models

The checkpoints for the pre-trained models are available [here]() as zip files. Copy them into the checkpoints folder (the default is ```./checkpoints```, 
create it if it doesn't yet exist) and unzip them. The folder structure should be  

You can generate images with a pre-trained checkpoint via ```test.py```:
```
python test.py --dataset_mode medical --name medical_pretrained \
--dataroot path_to/autopet
```
This script will create a folder named ```./results``` in which the resulting images are saved.

If you want to continue training from this checkpoint, use ```train.py``` with the same ```--name``` parameter and add ```--continue_train --which_iter best```.
## Citation
If you use this work please cite
```
@inproceedings{,
  title={},
  author={},
  booktitle={},
  year={2024}
}   
```
## Results

The generated images of our model are shown below: 
(From left to right, first images are labels, last images are ground_truth, the images in between are generated images with different random input noise):
![img.png](https://github.com/TWWinde/Medical-Images-Synthesis/blob/main/assert/combined_gerneated1.png)
This is the first edition of the model, which are not rewarding as the shape of the generated images vary a lot, the shape consistency is not 
good enough, especially at the boundary. So we pre-process the input images to remove artifacts from medical equipment(as shown in input pipeline above)
and use Mask Loss to enhance shape consistency. The basic idea is very straightforward and shown below.
![img.png](https://github.com/TWWinde/Medical-Images-Synthesis/blob/main/assert/maskloss.png)
After implementation:
![img.png](https://github.com/TWWinde/Medical-Images-Synthesis/blob/main/assert/combined_generated2.png)

### Ablation Study on AutoPET

| Exp   | Unpaired | Generator OASIS | Generator Wavelet | $\mathcal{L}_{mask}$ | FID   | LPIPS | SSIM   | RMSE | PSNR  |
|-------|----------|-----------------|-------------------|----------------------|-------|-------|--------|------|-------|
| Exp-1 |          | ✔️               |                   |                      | 15.83 | 0.27  | 0.9714 | 0.923| 15.39 |
| Exp-2 |          | ✔️               |                   | ✔️                    | **5.67** | 0.22  | 0.9713 | 0.45 | 19.83 |
| Exp-3 |          |                 | ✔️                 |                      | 7.29  | 0.06  | **0.9995** | **0.06** | **24.89** |
| Exp-4 |          |                 | ✔️                 | ✔️                    | 10.68 | **0.05** | 0.9995 | 0.06 | 23.27 |
| Exp-5 | ✔️        |                 | ✔️                 |                      | 10.76 | 0.26  | 0.9283 | 0.21 | 13.82 |

Legend:
- ✔️: Included in the experiment
- **Bold**: Best result in the column


### Ablation Study on SynthRAD2023

| Exp   | Unpaired | Generator OASIS | Generator Wavelet | $\mathcal{L}_{mask}$ | FID    | LPIPS | SSIM   | RMSE | PSNR  |
|-------|----------|-----------------|-------------------|----------------------|--------|-------|--------|------|-------|
| Exp-1 | ✔️       | ✔️              |                   |                      | 60.93  | 0.15  | 0.9983 | 0.12 | 18.54 |
| Exp-2 | ✔️       | ✔️              |                   | ✔️                   | 59.97  | 0.16  | 0.9980 | 0.12 | 18.09 |
| Exp-3 | ✔️       |                 | ✔️                |                      | **51.92** | 0.15  | 0.9983 | 0.12 | 18.69 |
| Exp-4 | ✔️       |                 | ✔️                | ✔️                   | 54.53  | **0.15** | 0.9984 | 0.12 | **18.77** |
| Exp-5 |          | ✔️              |                   | ✔️                   | 60.70  | 0.17  | 0.9978 | 0.13 | 17.69 |
| Exp-6 |          |                 | ✔️                | ✔️                   | 68.35  | 0.17  | **0.9985** | **0.11** | **19.08** |

Legend:
- ✔️: Feature is included in the experiment.
- **Bold**: Best score achieved in the column.

![img.png](https://github.com/TWWinde/Med-USIS/blob/main/images/maeimages.png)

### Ablation on 3D Noise

| 3D Noise Input | FID   | LPIPs | SSIM   | RMSE | PSNR  |
|----------------|-------|-------|--------|------|-------|
| w              | 51.92 | 0.15  | 0.9984 | 0.12 | 18.77 |
| w/o            | 60.35 | 0.26  | 0.9951 | 0.19 | 14.19 |


## License

This project is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.

For a list of other open source components included in this project, see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).

## Purpose of the project

This software is a research prototype, solely developed for and published as
part of the publication cited above. It will neither be
maintained nor monitored in any way.

## Contact
Please feel free to open an issue or contact us personally if you have questions, need help, or need explanations.
Write to one of the following email addresses, and maybe put one other in the cc:

twwinde@gmail.com  
st180408@stud.uni-stuttgart.de

