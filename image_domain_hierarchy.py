import os
from osgeo import gdal
import numpy as np
import torch
from torch.utils.data import Dataset
from consts import SPLIT_TRAIN
from image_domain_folder import ImageDomainFolder
import PIL.Image
from torchvision import transforms

class ImageDomainHierarchy(Dataset):

    def __init__(
        self, path, domain,
        split=SPLIT_TRAIN,
        transform=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self._path = os.path.join(path, split, domain)
        self._imgs = ImageDomainFolder.find_images_in_dir(self._path)
        self._transform = transform

    def __len__(self):
        return len(self._imgs)

    def __getitem__(self, index):
        path = self._imgs[index]
        result = PIL.Image.fromarray(self.loader_image(path))


        if self._transform is not None:
            result = self._transform(result)
        # result = np.array(result)  # Convert result to NumPy array
        # print('result.type',type(result))

        return result

    def loader_image(self, path):
        dataset = gdal.Open(path, gdal.GA_ReadOnly).ReadAsArray().astype('float32')
        if dataset is None:
            raise FileNotFoundError(f"Could not open {path}")
        if dataset.ndim == 3:
            return np.uint8(dataset.swapaxes(0,1).swapaxes(1,2))
        else:
            # return np.uint8(dataset)
            mat = np.uint8(dataset)[:,:,np.newaxis]
            return np.concatenate((mat,mat,mat),axis=2)
