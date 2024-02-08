#!/bin/bash

####

# Define slurm job parameters

####

#SBATCH --job-name=Med-USIS
#SBATCH --cpus-per-task=4
#SBATCH --partition=student
#SBATCH --gpus=1
#SBATCH --time 31-0
#SBATCH --nodelist=node-gpu-01
#SBATCH --error=job.%J.%x.err
#SBATCH --output=job.%J.%x.out


pyenv activate venv
module load cuda/11.3

#pre-process
python /home/students/studtangw1/Med-USIS/dataloaders/test.py



#experiment_1
#python train.py --name usis_wavelet --dataset_mode ct2mri --gpu_ids 0 \
#--dataroot /misc/data/private/autoPET/CT_MR --batch_size 4 --add_mask  \
#--netDu wavelet --continue_train \
#--model_supervision 0 --netG wavelet --channels_G 16  #16

#experiment_2
#python train.py --name usis_wavelet_no_mask --dataset_mode ct2mri --gpu_ids 0 \
#--dataroot /misc/data/private/autoPET/CT_MR --batch_size 4 \
#--netDu wavelet --continue_train \
#--model_supervision 0 --netG wavelet --channels_G 16  #16


#experiment_3.
#python train.py --name usis_oasis_generator --dataset_mode ct2mri --gpu_ids 0 \
#--dataroot /misc/data/private/autoPET/CT_MR --batch_size 2 --add_mask  \
#--netDu wavelet --continue_train \
#--model_supervision 0 --netG oasis --channels_G 64

#experiment_4
#python train.py --name usis_oasis_generator_no_mask --dataset_mode ct2mri --gpu_ids 0 \
#--dataroot /misc/data/private/autoPET/CT_MR --batch_size 2  \
#--netDu wavelet  --continue_train \
#--model_supervision 0 --netG oasis --channels_G 64

#experiment_5 unpaired ct autopet
#python train.py --name unpaired_ct_autopet --dataset_mode ct2ctautopet --gpu_ids 0 \
#--dataroot /misc/data/private/autoPET/data_nnunet --batch_size 4  \
#--netDu wavelet  \
#--model_supervision 0 --netG wavelet --channels_G 16

#experiment_6 no_3d_noise
#python train.py --name usis_wavelet --dataset_mode ct2mri --gpu_ids 0 \
#--dataroot /misc/data/private/autoPET/CT_MR --batch_size 4 --add_mask  \
#--netDu wavelet --continue_train --z_dim 0 --no_3dnoise\
#--model_supervision 0 --netG wavelet --channels_G 16  #16


#test
#python test.py --name usis_wavelet --dataset_mode ct2mri --gpu_ids 0 \
#--dataroot /misc/data/private/autoPET/CT_MR --batch_size 20 --model_supervision 0  \

