'''
Replace the Dataset in train_loop.py
'''

import torch
import random
import numpy as np
import os
import cv2
import nibabel as nib
import subprocess
import sys
import PIL.Image

subprocess.check_call([sys.executable, "-m", "pip", "install", "nibabel"])
def bbox2_3D(img):
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))
    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]
    return rmin, rmax, cmin, cmax, zmin, zmax

'''
    Randomly return an knee image slice on 
'''
class TKRDataset(torch.utils.data.Dataset):
    def __init__(self,
                 # you may have to add these params in config.json
                 data_dir: str,
                 # Plane: 0-saggital-slice of first dimension, 1-coronal-slice on second dimension
                 plane: int, 
                 with_constraint: False,
                 slice_delta_on_each_side: int = 30,
                 c_size: int=64,
                 
                 ):
        self.c_size = c_size
        self.data_dir = data_dir
        folders = os.listdir(data_dir)
        assert 'TKR' in folders and 'NO_TKR' in folders
        self.tkr_samples = os.listdir(os.path.join(data_dir, 'TKR'))
        self.non_tkr_samples = os.listdir(os.path.join(data_dir, 'NO_TKR'))
        self.plane = plane
        self.delta = slice_delta_on_each_side
        random.shuffle(self.tkr_samples)
        random.shuffle(self.non_tkr_samples)
        self.num_channels = 1
        super().__init__()


    def __len__(self):
        return len(self.tkr_samples) + len(self.non_tkr_samples)

    @property
    def image_shape(self):
        return self.__getitem__(0)[0].shape

    @property
    def label_dim(self):
        return 512

    @property
    def resolution(self):
        #was (1, 256, 256) in oai dataset
        #now only (160, 384)
        assert len(self.image_shape) == 3
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    def __getitem__(self, idx):
        if random.random() > 0.5:
            sample = random.choice(self.tkr_samples)
            target = 0
            img = nib.load(os.path.join(self.data_dir, 'TKR', sample)).get_fdata()
        else:
            sample = random.choice(self.non_tkr_samples)
            target = 1
            img = nib.load(os.path.join(self.data_dir, 'NO_TKR', sample)).get_fdata()

        if random.random() > 0.5:
            # 50% chance left right flip
            img = np.flip(img, 0)

        x, y, z = img.shape
        if self.plane == 1:
            mid = y // 2
            random_index = random.choice(range(mid-self.delta, mid+self.delta))
            img_s = img[:, random_index, :]
            img_s = cv2.normalize(img_s, None, 0,255, cv2.NORM_MINMAX)
            img_s=cv2.resize(img_s, (256, 256))
        elif self.plane == 0:
            mid = x // 2
            random_index = random.choice(range(mid-self.delta, mid+self.delta))
            img_s = img[random_index, :, :]
            img_s = cv2.normalize(img_s, None, 0, 255, cv2.NORM_MINMAX)
            img_s=cv2.resize(img_s, (256, 256))

        return img_s, target
