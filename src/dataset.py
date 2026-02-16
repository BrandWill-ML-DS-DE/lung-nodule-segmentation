import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset

class LungDataset(Dataset):
    def __init__(self, image_dir, mask_dir, patch_size=(64,64,64)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.patch_size = patch_size
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def random_patch(self, img, mask):
        z, y, x = img.shape
        pz, py, px = self.patch_size

        z1 = np.random.randint(0, z - pz)
        y1 = np.random.randint(0, y - py)
        x1 = np.random.randint(0, x - px)

        return (
            img[z1:z1+pz, y1:y1+py, x1:x1+px],
            mask[z1:z1+pz, y1:y1+py, x1:x1+px]
        )

    def __getitem__(self, idx):
        img = nib.load(os.path.join(self.image_dir, self.images[idx])).get_fdata()
        mask = nib.load(os.path.join(self.mask_dir, self.images[idx])).get_fdata()

        img_patch, mask_patch = self.random_patch(img, mask)

        img_patch = torch.tensor(img_patch).unsqueeze(0).float()
        mask_patch = torch.tensor(mask_patch).unsqueeze(0).float()

        return img_patch, mask_patch

