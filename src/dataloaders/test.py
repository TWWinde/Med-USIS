from PIL import Image
from matplotlib import pylab as plt
import nibabel as nib
import numpy as np
import os
from torchvision import transforms as TR
from torchvision.transforms import functional

filename1 = '/mnt/qdata/rawdata/NAKO_706/NAKO_706_nifti/3D_GRE_TRA_W_COMPOSED/100000_30/fat.nii.gz'
filename2 = '/mnt/qdata/rawdata/NAKO_706/NAKO_706_nifti/3D_GRE_TRA_W_COMPOSED/100000_30/in.nii.gz'
filename3 = '/mnt/qdata/rawdata/NAKO_706/NAKO_706_nifti/3D_GRE_TRA_W_COMPOSED/100000_30/opp.nii.gz'
filename4 = '/mnt/qdata/rawdata/NAKO_706/NAKO_706_nifti/3D_GRE_TRA_W_COMPOSED/100000_30/water.nii.gz'
filename5 = '/mnt/qdata/rawdata/NAKO_706/NAKO_706_nifti/PD_FS_SPC_COR/100005_30/17_pd_fs_spc_cor.nii.gz'
data_path = '/home/students/studtangw1/data'
path = [filename1,filename2,filename3,filename4,filename5]



for i in path:
    img1 = nib.load(path)
    img_3d = img1.get_fdata()
    print(img_3d.shape)

    for z in range(img_3d.shape[2]):
        img_slice = img_3d[:, :, z]
        if img_slice.max() != img_slice.min():
            image = (((img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())) * 255).astype(np.uint8)
            image = Image.fromarray(image)
            image = image.convert('RGB')
            image.save(f'/Users/tangwenwu/Desktop/thesis/data/train/CT/CT_slice_{z}.png')


