"""
Copyright (c) Hitachi, Ltd. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the CDB-loss/CIFAR-LT directory.
=======
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import random
import pickle
import torchvision
import json
from torchvision import transforms
from torch.autograd import Variable
from models import *
from data import *


def build_model(class_num):
    model = ResNet32(class_num)

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True

    return model



def test(model, dataloaders, args):
   
    testloader = dataloaders['test']
    mwnet = VNet(100, 128, 128, 100).cuda()
    mwnet.load_state_dict(torch.load('saved_model/diff2weight_cifar100_imbalance100_exaugment_lambda1.0_no_metaloss.pth'))
    #mwnet.linear3.bias = torch.zeros(100)
    cls_id = np.arange(100)
    freq_max = 500
    cls_freq = freq_max * (1/args.imbalance) ** (cls_id / 99)
    
    model.eval()
    mwnet.eval()
    test_total = 0
    test_correct = 0
    class_wise_accuracy = np.zeros(100)

    
    for test_images, test_labels in testloader:
          test_images = test_images.cuda()
          test_labels = test_labels.type(torch.cuda.LongTensor)
          with torch.no_grad():
               out = model(test_images)
               #out = out/torch.norm(out, 2, 1, keepdim=True)
               #print(out.norm(2,1))
               #out = torch.mm(out, ws.t())
          _, predicted = out.max(1)
          test_total += test_labels.size(0)
          test_correct += predicted.eq(test_labels).sum().item()
          for id in range(len(predicted)): 
             if predicted[id] == test_labels[id]:
                 class_wise_accuracy[int(predicted[id])] += 1/100
    #accuracy = 1 * np.ones(100)
    #accuracy[0] = 0.8
    #accuracy[-1] = 0.8

    print('Test Accuracy for is %.4f'%(test_correct/ test_total))
    print('Many-shot test accuracy is %.4f'%(class_wise_accuracy[cls_freq>200].mean()))
    print('Med-shot test accuracy is %.4f'%(class_wise_accuracy[(cls_freq<=200)&(cls_freq>=100)].mean()))
    print('Few-shot test accuracy is %.4f'%(class_wise_accuracy[cls_freq<100].mean()))


    


def main():
   parser = argparse.ArgumentParser(description='Input parameters for CIFAR-test set testing')
   parser.add_argument('--saved_model_path', type=str, default='./saved_model/best_cifar100_resnet32_loss_CE_imbalance100_exaugment_lambda1.0_no_metaloss.pth', help='model path for testing')
   parser.add_argument('--class_num', type=int, default=100, help='number of classes')
   parser.add_argument('--imbalance', type=int, default=100, help='imbalance value')
   parser.add_argument('--n_gpus', type=int, default=4, help='number of gpus to use')
   args = parser.parse_args()
   model = build_model(args.class_num)
   #model = resnet.resnet32()
   #model.linear = nn.Linear(in_features=64, out_features=args.class_num, bias=True)
   #model = nn.DataParallel(model, device_ids=range(args.n_gpus))
   model = model.cuda()
   model.load_state_dict(torch.load(args.saved_model_path))
   #print(model.module.linear.bias.data.unsqueeze(1).norm(1,1))
   #model.linear.bias.data = torch.zeros(100).cuda()
   
   ## Load test data and labels
   
   test_images, test_labels, class_names = load_test_data(args)
   
   #model = weight_scaling(model)

   ##Loading done
             
   test_images = test_images.reshape(-1, 3, 32, 32)  ## reshape to N, C, H, W
   #normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
   #                                  std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
   normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
            std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
   transform_test = transforms.Compose([
       transforms.ToTensor(),
       normalize,
   ])
   test_set = createCIFAR(test_images, test_labels, transforms=transform_test)
   testloader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=8)
   dataloaders = {'test': testloader}
   test(model, dataloaders, args)

if __name__ == '__main__':
   main()

   
