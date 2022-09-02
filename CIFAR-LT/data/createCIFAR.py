"""
Copyright (c) Hitachi, Ltd. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the CDB-loss/CIFAR-LT directory.
=======
"""

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class createCIFAR(Dataset):
    def __init__(self, mat, labels, transforms, pairwise=False):
        self.mat = mat
        self.labels = labels
        self.transforms = transforms
        self.pairwise = pairwise
        if self.pairwise:
            self.labels_set = set(self.labels)
            self.label_to_indices = {label: np.where(self.labels == label)[0]
                                     for label in self.labels_set}
    
    def image_process(self, image):
        image = np.transpose(image,(1,2,0))
        image = Image.fromarray(np.uint8(image))
        image = self.transforms(image).float()
        return image

    def __getitem__(self, item):

        if self.pairwise:
            target = np.random.randint(0,2)
            img1, label1 = self.image_process(self.mat[item % len(self.labels)]), self.labels[item % len(self.labels)]
            if target == 1:
                siamese_index = item
                while siamese_index == item:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.image_process(self.mat[siamese_index])
            return (img1, img2), target
        else:
            image = self.mat[item % len(self.labels)]
            label = self.labels[item % len(self.labels)]
            #image = np.reshape(image, (224, 224))
            image = np.transpose(image,(1,2,0))
            image = Image.fromarray(np.uint8(image))
            image = self.transforms(image).float()
        return image, label
    def __len__(self):
        return len(self.labels)
    #def _get_label(self, item):
    #    return self.labels[item % len(self.labels)]
