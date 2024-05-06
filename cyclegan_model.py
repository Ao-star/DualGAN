import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from torchvision.datasets.folder import IMG_EXTENSIONS
from consts import SPLIT_TRAIN
from osgeo import gdal
import numpy as np
import PIL.Image
from torchvision import transforms

# 定义数据集
class ImageDomainFolder(Dataset):
    """Dataset structure introduced in a CycleGAN paper.

    This dataset expects images to be arranged into subdirectories
    under `path`: `trainA`, `trainB`, `testA`, `testB`. Here, `trainA`
    subdirectory contains training images from domain "a", `trainB`
    subdirectory contains training images from domain "b", and so on.

    Parameters
    ----------
    path : str
        Path where the dataset is located.
    domain : str
        Choices: 'a', 'b'.
    split : str
        Choices: 'train', 'test', 'val'
    transform : Callable or None,
        Optional transformation to apply to images.
        E.g. torchvision.transforms.RandomCrop.
        Default: None
    """

    def __init__(
        self, path,
        domain        = 'a',
        split         = SPLIT_TRAIN,
        transform     = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        subdir = split + domain
        self._path      = os.path.join(path, subdir)
        self._imgs      = ImageDomainFolder.find_images_in_dir(self._path)
        self._transform = transform

    @staticmethod
    def find_images_in_dir(path):
        extensions = set(IMG_EXTENSIONS)

        result = []
        for fname in os.listdir(path):
            fullpath = os.path.join(path, fname)

            if not os.path.isfile(fullpath):
                continue

            ext = os.path.splitext(fname)[1]
            if ext not in extensions:
                continue

            result.append(fullpath)

        result.sort()
        return result

    def __len__(self):
        return len(self._imgs)

    def __getitem__(self, index):
        path   = self._imgs[index]
        result = self.loader_image(path)

        if self._transform is not None:
            result = self._transform(result)
        return result
        
    def loader_image(self, path):
        dataset = gdal.Open(path, gdal.GA_ReadOnly).ReadAsArray().astype('float32')
        if dataset is None:
            raise FileNotFoundError(f"Could not open {path}")

        print(torch.from_numpy(dataset))
        return torch.from_numpy(dataset)


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
        
# 定义生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # 定义网络结构
        ...

    def forward(self, x):
        # 实现前向传播
        ...


# 定义判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # 定义网络结构
        ...

    def forward(self, x):
        # 实现前向传播
        ...


# 定义CycleGAN模型
class CycleGAN(nn.Module):
    def __init__(self):
        super(CycleGAN, self).__init__()

        self.generator_A2B = Generator()
        self.generator_B2A = Generator()
        self.discriminator_A = Discriminator()
        self.discriminator_B = Discriminator()

    def forward(self, real_A, real_B):
        # 实现前向传播
        ...

    def backward(self):
        # 实现反向传播
        ...


# 训练模型
def train():
    dataset_A = CycleGANDataset('path/to/dataset_A')
    dataset_B = CycleGANDataset('path/to/dataset_B')
    dataloader_A = DataLoader(dataset_A, batch_size=batch_size, shuffle=True)
    dataloader_B = DataLoader(dataset_B, batch_size=batch_size, shuffle=True)

    model = CycleGAN()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        for i, (real_A, real_B) in enumerate(zip(dataloader_A, dataloader_B)):
            fake_A = model.generator_B2A(real_B)
            fake_B = model.generator_A2B(real_A)

            # 计算判别器的损失
            pred_real_A = model.discriminator_A(real_A)
            pred_fake_A = model.discriminator_A(fake_A.detach())
            loss_discriminator_A = criterion(pred_real_A, torch.ones_like(pred_real_A)) + criterion(pred_fake_A, torch.zeros_like(pred_fake_A))

            pred_real_B = model.discriminator_B(real_B)
            pred_fake_B = model.discriminator_B(fake_B.detach())
            loss_discriminator_B = criterion(pred_real_B, torch.ones_like(pred_real_B)) + criterion(pred_fake_B, torch.zeros_like(pred_fake_B))

            loss_discriminator = 0.5 * (loss_discriminator_A + loss_discriminator_B)

            # 计算生成器的损失
            pred_fake_A = model.discriminator_A(fake_A)
            pred_fake_B = model.discriminator_B(fake_B)
            loss_generator_A2B = criterion(pred_fake_B, torch.ones_like(pred_fake_B))
            loss_generator_B2A = criterion(pred_fake_A, torch.ones_like(pred_fake_A))

            # 计算循环一致性损失
            recovered_A = model.generator_B2A(fake_B)
            recovered_B = model.generator_A2B(fake_A)
            loss_cycle_A = criterion(recovered_A, real_A)
            loss_cycle_B = criterion(recovered_B, real_B)

            loss_generator = loss_generator_A2B + loss_generator_B2A + 10 * (loss_cycle_A + loss_cycle_B)

            # 更新网络权重
            optimizer.zero_grad()
            loss_discriminator.backward(retain_graph=True)
            loss_generator.backward()
            optimizer.step()
