import os
import logging
import torch
import scipy.io
import numpy as np
from os import listdir
from os.path import splitext
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(self, patch_dir):
        self.patch_dir = patch_dir
        self.ids = [splitext(file)[0] for file in listdir(patch_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def preprocess(self, pil_img):
        if len(pil_img.shape) == 2:
            pil_img = np.expand_dims(pil_img, axis=2)
        # HWC to CHW
        img_trans = pil_img.transpose((2, 0, 1))
        return img_trans.astype(float)

    def __getitem__(self, i):
        idx = self.ids[i]
        mat = scipy.io.loadmat(os.path.join(self.patch_dir, idx+'.mat'))
        img_arr = mat['image']
        mask_arr = mat['mask']
        img_arr = self.preprocess(img_arr)
        mask_arr = self.preprocess(mask_arr)
        return {'image': torch.from_numpy(img_arr), 'mask': torch.from_numpy(mask_arr)}
