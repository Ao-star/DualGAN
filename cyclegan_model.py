import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from osgeo import gdal
from torchvision import transforms

# 定义数据集类
class ImageDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_files = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        img = self.loader_image(img_path)
        return img

    def loader_image(self, path):
        dataset = gdal.Open(path, gdal.GA_ReadOnly).ReadAsArray().astype('float32')
        if dataset is None:
            raise FileNotFoundError(f"Could not open {path}")
        if dataset.ndim == 3:
            return np.uint8(dataset.swapaxes(0,1).swapaxes(1,2))
        else:
            mat = np.uint8(dataset)[:,:,np.newaxis]
            return np.concatenate((mat,mat,mat),axis=2)
          
        return torch.from_numpy(data)

# 定义生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # 定义生成器的结构

    def forward(self, x):
        # 定义生成器的前向传播逻辑
        return x

# 定义判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 定义判别器的结构

    def forward(self, x):
        # 定义判别器的前向传播逻辑
        return x

# 初始化生成器和判别器
G_AB = Generator()
G_BA = Generator()
D_A = Discriminator()
D_B = Discriminator()

# 定义损失函数和优化器
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
optimizer_G = optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_A = optim.Adam(D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_B = optim.Adam(D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 定义训练循环
def train_cycle_gan(dataset_A_path, dataset_B_path, output_path, num_epochs=200, batch_size=1):
    # 创建数据加载器
    dataset_A = ImageDataset(dataset_A_path)
    dataset_B = ImageDataset(dataset_B_path)
    dataloader_A = DataLoader(dataset_A, batch_size=batch_size, shuffle=True)
    dataloader_B = DataLoader(dataset_B, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        for real_A, real_B in zip(dataloader_A, dataloader_B):
            # 训练生成器和判别器
          
            # 保存模型和生成的样本
            if (epoch + 1) % 10 == 0:
                torch.save(G_AB.state_dict(), os.path.join(output_path, f'G_AB_epoch{epoch + 1}.pth'))
                torch.save(G_BA.state_dict(), os.path.join(output_path, f'G_BA_epoch{epoch + 1}.pth'))
                # 生成一些样本并保存

# 调用训练函数
dataset_A_path = "/run/media/teamshare/workspace_bingo05/uvcgan2-main/data/patch_data/train/source/"
dataset_B_path = "/run/media/teamshare/workspace_bingo05/uvcgan2-main/data/patch_data/train/target/"
output_path = "/run/media/teamshare/workspace_bingo05/uvcgan2-main/outdir/result/"
train_cycle_gan(dataset_A_path, dataset_B_path, output_path)
