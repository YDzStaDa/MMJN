import numpy as np
import glob
import torch.utils.data
import os
import math
from skimage import io, transform
from PIL import Image
import torch
import torchvision as vision
from torchvision import transforms, datasets
import random

class MMJDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, scale_aug=False, rot_aug=False, test_mode=False, num_models=0, num_views=5, shuffle=True):
        self.classnames=[]
        self.root_dir = root_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        self.num_views = num_views
        parent_dir = root_dir.rsplit('/')[0]
        self.filepaths = []
        for i in range(len(self.classnames)):
            all_files = sorted(glob.glob(parent_dir+'name'))
            stride = int(5/self.num_views)
            all_files = all_files[::stride]
            if num_models == 0:
                self.filepaths.extend(all_files)
            else:
                self.filepaths.extend(all_files[:min(num_models,len(all_files))])

        if shuffle==True:
            rand_idx = np.random.permutation(int(len(self.filepaths)/num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.filepaths[rand_idx[i]*num_views:(rand_idx[i]+1)*num_views])
            self.filepaths = filepaths_new

        if self.test_mode:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return int(len(self.filepaths)/self.num_views)

    def __getitem__(self, idx):
        path = self.filepaths[idx*self.num_views]
        class_name = path.split('/')
        class_id = self.classnames.index(class_name)
        imgs = []
        for i in range(self.num_views):
            im = Image.open(self.filepaths[idx*self.num_views+i]).convert('')
            if self.transform:
                im = self.transform(im)
            imgs.append(im)
        return (class_id, torch.stack(imgs), self.filepaths[idx*self.num_views:(idx+1)*self.num_views])

