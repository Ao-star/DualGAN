
import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from osgeo import gdal

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)

        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.tif"))

    def gdal_loader(self, path):
        dataset = gdal.Open(path, gdal.GA_ReadOnly)
        if dataset is None:
            raise FileNotFoundError(f"Could not open {path}")
        array = dataset.ReadAsArray().astype('float32')
        if array.ndim == 3:
            return np.uint8(array.swapaxes(0, 1).swapaxes(1, 2))
        else:
            mat = np.uint8(array)[:, :, np.newaxis]
            return np.concatenate((mat, mat, mat), axis=2)

    def __getitem__(self, index):
        img = self.gdal_loader(self.files[index % len(self.files)])
        h, w, _ = img.shape
        img_A = img[:, :w // 2, :]
        img_B = img[:, w // 2:, :]

        if np.random.random() < 0.5:
            img_A = img_A[:, ::-1, :]
            img_B = img_B[:, ::-1, :]

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return len(self.files)

