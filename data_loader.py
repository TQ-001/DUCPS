import os
import torch
# from skimage import io, transform
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd

class Singledataset(Dataset):
    """LUS spatial and radon dataset."""

    def __init__(self ,  img_dir,  transform=None ):
        self.img_dir = img_dir
        self.image_list = [x for x in os.listdir(img_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.image_list[idx])
        image =  Image.open(img_name)
        # if len(image.shape) == 3:
        #     image = image.unsqueeze(0)
        if self.transform:
            image = self.transform(image)
        # if len(target.shape) == 3:
        #     target = target.unsqueeze(0)

        return image, os.path.splitext(self.image_list[idx])[0]
    

class SingledatasetLISTA(Dataset):
    """LUS spatial and radon dataset."""

    def __init__(self ,  img_dir,  transform=None ):
        self.img_dir = img_dir
        self.image_list = [x for x in os.listdir(img_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.image_list[idx])
        image =  Image.open(img_name)
        # if len(image.shape) == 3:
        #     image = image.unsqueeze(0)
        if self.transform:
            image = self.transform(image)
        # if len(target.shape) == 3:
        #     target = target.unsqueeze(0)

        return image,image
    

class SRdataset(Dataset):
    """LUS spatial and radon dataset."""

    def __init__(self , spatial_img_dir, radon_img_dir, spatial_transform=None , radon_transform=None):
        self.spatial_img_dir = spatial_img_dir
        self.radon_img_dir = radon_img_dir
        self.spatial_image_list = [x for x in os.listdir(spatial_img_dir)]
        self.radon_image_list = [x for x in os.listdir(radon_img_dir)]
        self.spatial_transform = spatial_transform
        self.radon_transform = radon_transform

    def __len__(self):
        return len(self.spatial_image_list)

    def __getitem__(self, idx):
        spatial_img_name = os.path.join(self.spatial_img_dir, self.spatial_image_list[idx])
        spatial_image =  Image.open(spatial_img_name)
        radon_img_name = os.path.join(self.radon_img_dir, self.radon_image_list[idx])
        radon_image =  Image.open(radon_img_name)
        # if len(image.shape) == 3:
        #     image = image.unsqueeze(0)
        if self.spatial_transform:
            spatial_image = self.spatial_transform(spatial_image)
        # if len(target.shape) == 3:
        #     target = target.unsqueeze(0)
        if self.radon_transform:
            radon_image = self.radon_transform(radon_image)

        return spatial_image, radon_image, os.path.splitext(self.spatial_image_list[idx])[0]

class Radondataset(Dataset):
    """LUS radon dataset for training."""

    def __init__(self ,  img_dir,  transform=None ):
        self.img_dir_1 = img_dir+'1'
        self.img_dir_2 = img_dir+'2'
        self.image_list_1 = [x for x in os.listdir(self.img_dir_1)]
        self.image_list_2 = [x for x in os.listdir(self.img_dir_2)]
        self.transform = transform

    def __len__(self):
        return len(self.image_list_1)

    def __getitem__(self, idx):

        img_name_1 = os.path.join(self.img_dir_1, self.image_list_1[idx])
        image_1 =  Image.open(img_name_1)
        img_name_2 = os.path.join(self.img_dir_2, self.image_list_2[idx])
        image_2 =  Image.open(img_name_2)
        # if len(image.shape) == 3:
        #     image = image.unsqueeze(0)
        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)

        # if len(target.shape) == 3:
        #     target = target.unsqueeze(0)

        return image_1,image_2, os.path.splitext(self.image_list_1[idx])[0]