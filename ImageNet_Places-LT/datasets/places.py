"""
Copyright (c) Hitachi, Ltd. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the MiSLAS project at https://github.com/dvlab-research/MiSLAS
"""

import os
import numpy as np
from PIL import Image

import torch
import torchvision
import torchvision.datasets
from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
from .sampler import ClassAwareSampler


def sep_meta_val(txt):
    num_classes = 365
    meta_img_path = []
    meta_targets = []
    val_img_path = []
    val_targets = []
    meta_data = np.zeros(num_classes)
    with open(txt) as f:
        for line in f:
            path = line.split()[0]
            if 'val/' in path:
                home, _, img = path.split('/')
                path = os.path.join(home, img)
            target = int(line.split()[1])
            if meta_data[target] < 10:
                meta_img_path.append(line.split()[0])
                meta_targets.append(target)
                meta_data[target] += 1
            else:
                val_img_path.append(os.path.join(line.split()[0]))
                val_targets.append(int(line.split()[1]))
        # val_targets = np.array(class_map)[val_targets].tolist()
        # meta_targets = np.array(class_map)[meta_targets].tolist()
    return val_img_path, val_targets, meta_img_path, meta_targets



class LT_Dataset(Dataset):
    num_classes = 365

    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.targets = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.targets.append(int(line.split()[1]))
        
        cls_num_list_old = [np.sum(np.array(self.targets) == i) for i in range(self.num_classes)]
        
        # generate class_map: class index sort by num (descending)
        sorted_classes = np.argsort(-np.array(cls_num_list_old))
        self.class_map = [0 for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.class_map[sorted_classes[i]] = i
        
        self.targets = np.array(self.class_map)[self.targets].tolist()
        
        

        self.class_data = [[] for i in range(self.num_classes)]
        for i in range(len(self.targets)):
            j = self.targets[i]
            self.class_data[j].append(i)

        self.cls_num_list = [np.sum(np.array(self.targets)==i) for i in range(self.num_classes)]


    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        path = self.img_path[index]
        target = self.targets[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


class LT_Dataset_SepVal(Dataset):
    num_classes = 365

    def __init__(self, root, img_paths, targets, class_map, transform=None):
        self.img_path = img_paths
        self.targets = targets
        self.transform = transform
        self.class_map = class_map
        self.root = root
        self.targets = np.array(self.class_map)[self.targets].tolist()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        path = os.path.join(self.root, self.img_path[index])
        target = self.targets[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target



class LT_Dataset_Eval(Dataset):
    num_classes = 365

    def __init__(self, root, txt, class_map, transform=None):
        self.img_path = []
        new_img_path = np.array([])
        self.targets = []
        new_targets = []
        self.transform = transform
        self.class_map = class_map
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.targets.append(int(line.split()[1]))

        self.targets = np.array(self.class_map)[self.targets].tolist()

        

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        path = self.img_path[index]
        target = self.targets[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


class LT_Dataset_Val(Dataset):
    num_classes = 365

    def __init__(self, root, txt, class_map, transform=None):
        self.img_path = []
        new_img_path = np.array([])
        self.targets = []
        new_targets = []
        self.transform = transform
        self.class_map = class_map
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.targets.append(int(line.split()[1]))

        self.targets = np.array(self.class_map)[self.targets]#.tolist()
        
        self.img_path = np.array(self.img_path)
        for target in np.unique(self.targets):
             idx = np.where(self.targets == target)[0]
             
             select = np.random.choice(idx, 10, replace=False)       ### sampling 10 samples per class as meta/val
             new_img_path = np.concatenate([new_img_path, self.img_path[select]])
             new_targets = np.concatenate([new_targets, self.targets[select]])
        self.img_path = new_img_path.tolist()
        self.targets = new_targets.astype(int).tolist()
        

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        path = self.img_path[index]
        target = self.targets[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


class Places_LT(object):
    def __init__(self, distributed, root="", batch_size=60, num_works=40, reuse_val_as_meta=True):
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0), (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            normalize,
            ])
        

        transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        
        train_txt = "./datasets/data_txt/Places_LT_train.txt"
        val_txt = "./datasets/data_txt/Places_LT_val.txt"
        eval_txt = "./datasets/data_txt/Places_LT_test.txt"

        
        train_dataset = LT_Dataset(root, train_txt, transform=transform_train)
        # val_dataset = LT_Dataset_Eval(root, val_txt, transform=transform_test, class_map=train_dataset.class_map)
        eval_dataset = LT_Dataset_Eval(root, eval_txt, transform=transform_test, class_map=train_dataset.class_map)
        
        self.cls_num_list = train_dataset.cls_num_list

        if reuse_val_as_meta:
            val_dataset = LT_Dataset_Val(root, val_txt, transform=transform_test, class_map=train_dataset.class_map)
            meta_dataset = val_dataset
        else:
            val_images, val_labels, meta_images, meta_labels = sep_meta_val(val_txt)
            val_dataset = LT_Dataset_SepVal(root, val_images, val_labels, transform=transform_test, class_map=train_dataset.class_map)
            meta_dataset = LT_Dataset_SepVal(root, meta_images, meta_labels, transform=transform_test,
                                            class_map=train_dataset.class_map)

        self.dist_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed else None
        self.train_instance = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_works, pin_memory=True, sampler=self.dist_sampler)

        balance_sampler = ClassAwareSampler(train_dataset)
        self.train_balance = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_works, pin_memory=True, sampler=balance_sampler)
        self.validate = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=128, shuffle=True,
            num_workers=num_works, pin_memory=True)

        self.meta = torch.utils.data.DataLoader(
            meta_dataset,
            batch_size=100, shuffle=True,
            num_workers=num_works, pin_memory=True)

        self.eval = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_works, pin_memory=True)
