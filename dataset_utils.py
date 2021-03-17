import torch
from torch import nn
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from skimage import io, transform
import torchvision
import torchvision.transforms.functional as F


class ImpressionismDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.metadata = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.metadata['new_filename'][idx])
        
        image = None
        if os.path.exists(img_name):
            image = io.imread(img_name)
        else:
            image = np.random.randint(1, 255, (128,128,3))
            
        label = self.metadata['label'][idx]
        label = np.array([label])
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    
################## Customs tranforms ###########################

class ImpressionismToTensor(torchvision.transforms.ToTensor):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, pic):
        image, label = pic['image'], pic['label']

        return {'image': F.to_tensor(image),
                'label': torch.from_numpy(label)}

    
class ImpressionismNormalize(torchvision.transforms.Normalize):
    """Convert ndarrays in sample to Tensors."""

    def forward(self, sample):
        image, label = sample['image'], sample['label']

        return {'image': F.normalize(image, self.mean, self.std, self.inplace),
                'label': label}

class ImpressionismResize(torchvision.transforms.Resize):
    
    def forward(self, sample):
        #Unpack the image and label
        image, label = sample['image'], sample['label']
        
        return {'image': F.resize(image, self.size, self.interpolation),
                'label': label}
   
