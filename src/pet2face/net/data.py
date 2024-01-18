from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
from PIL import Image
import torch
from torchvision import transforms, utils
from typing import List

##dataset class
class ImageDataset(Dataset):
  def __init__(self, 
                list_ct:List[str], 
                list_pet:List[str], 
                transform=None):
    self.ct_names = list_ct
    self.pet_names = list_pet
    self.transform = transform

  def __len__(self):
    return(len(self.ct_names))

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    ct = np.array(Image.open(self.ct_names[idx]), dtype=np.float32)
    pet = np.array(Image.open(self.pet_names[idx]), dtype=np.float32)
    sample = {"pet":pet, "ct":ct}

    if self.transform:
      sample = self.transform(sample)
    return sample

## transforms data_train
class Crop(object):
  def __call__(self, sample, cs=152):
    pet, ct = sample["pet"], sample["ct"]
    return {"pet": pet[cs:-cs, cs:-cs], "ct": ct[cs:-cs, cs:-cs]}

class Normalize(object):
  def __call__(self, sample):
    pet, ct = sample['pet'], sample['ct']
    pet /= 255
    ct /= 255
    return {"pet": pet, "ct": ct}

class ToTensor(object):
  def __call__(self, sample):
    pet, ct = sample["pet"], sample["ct"]
    # transpose imgs because 
    # np img : HxWxC
    # torch img : CxHxW
    return {"pet": torch.Tensor(pet).unsqueeze(0), "ct":torch.Tensor(ct).unsqueeze(0)}