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
from torch.utils.data import DataLoader
import sys
import argparse
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import models
from models import *
from losses import *
from data import *
from utils import *
import importlib
import logging
from auto_augment import AutoAugment, Cutout
#from autoaug import CIFAR10Policy, Cutout
from torch.nn.parameter import Parameter
from methods import LabelAwareSmoothing, LearnableWeightScaling




logging.basicConfig(level=logging.INFO, filename='cifar-100_imbalance200_mwnet_stage2.log')


class DotProduct_Classifier(nn.Module,):
    
    def __init__(self, num_classes=100, feat_dim=64, *args):
        super(DotProduct_Classifier, self).__init__()
        # print('<DotProductClassifier> contains bias: {}'.format(bias))
        self.fc = nn.Linear(feat_dim, num_classes, bias=True)
        self.scales = Parameter(torch.ones(num_classes), requires_grad=True) #make required_grad=False if CRT
        for param_name, param in self.fc.named_parameters():
            param.requires_grad = True #make False if LWS
        
        
    def forward(self, x, *args):
        x = self.fc(x)
        x *= self.scales
        return x

def build_model(class_num):
    model = ResNet32_ft(class_num)

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True

    return model

def adjust_learning_rate(optimizer, epochs):
    #lr = 0.1  * ((0.1 ** int(epochs >= 20)) * (0.1 ** int(epochs >= 30)))
    #lr = (0.1 + ((0.1 * epochs/400) * int(epochs <=100)) + (0.1 * int(epochs > 100)))  * ((0.1 ** int(epochs >=300)) * (0.1 ** int(epochs >= 500)))  # For WRN-28-10
    #lr = 0.1 * ((0.1 ** int(epochs >= 6400)) * (0.1 ** int(epochs >= 9600)))
    lr = 0.5 * (0.05 - 0) * (1 + math.cos(epochs / 50 * 3.1415926535))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def source_import(file_path):
    """This function imports python module directly from source code using importlib"""
    spec = importlib.util.spec_from_file_location('', file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def train(model, dataloaders, args):
   
   #params_dict = dict(model.named_params(model))
   #print(params_dict.items())
   params = []

   

   ## define optimizer and scheduler

   #tau = torch.nn.Parameter(torch.cuda.FloatTensor([1]), requires_grad=True)
   classifier = DotProduct_Classifier().cuda()
   classifier.fc.weight.data = model.linear.weight.data
   classifier.fc.bias.data = model.linear.bias.data
   #params = [{'params': tau, }]
   #tau = tau.cuda()
   optimizer = torch.optim.SGD(classifier.parameters(), lr=0.05, weight_decay=1e-4, momentum=0.9)
   #optimizer = torch.optim.Adam(params, lr=1e-8,)

   #optimizer = torch.optim.SGD(params, momentum=0.9)
   # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[160, 210], gamma=0.01)
   #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=240)
   weights = model.linear.weight.data
   norms = weights.norm(2, 1).unsqueeze(0).cuda()
   ## create loss
   
   if args.stage_2_method == 'LAS':
       if args.imbalance > 100:
           smooth_head, smooth_tail = 0.4, 0.0
       elif args.imbalance > 50:
           smooth_head, smooth_tail = 0.4, 0.1
       elif args.imbalance > 20:
           smooth_head, smooth_tail = 0.3, 0
       else:
           smooth_head, smooth_tail = 0.2, 0
       loss = LabelAwareSmoothing(cls_num_list=args.cls_num_list, smooth_head=smooth_head,
                           smooth_tail=smooth_tail).cuda()
   else:
       loss = torch.nn.CrossEntropyLoss(reduction='none').cuda()


   ## create folder for saving model
      
   os.makedirs(args.save_model, exist_ok = True)
   
   trainloader = dataloaders['train']
   valloader = dataloaders['val']
   metaloader = dataloaders['meta']
   best_val_accuracy = 0




   for epoch in range(args.max_epochs):
      epoch_loss = 0
      total_train_samples = 0

      adjust_learning_rate(optimizer, epoch)

      for batch_i, (train_images, train_labels) in enumerate(trainloader):
         #model.train()
         model.eval()
         #mwnet.train()
         train_images = Variable(train_images.cuda(), requires_grad=False)
         train_labels = Variable(train_labels.type(torch.cuda.LongTensor), requires_grad=False)

         optimizer.zero_grad()
         with torch.no_grad():
            out= model(train_images)
         out = classifier(out)
         train_loss = loss(out, train_labels) 
         epoch_loss += torch.sum(train_loss).item()
         total_train_samples += len(train_labels) 
         #train_loss = torch.mean(train_loss * weights)
         train_loss = torch.mean(train_loss)
         train_loss.backward()
         optimizer.step()
      logging.info('Epoch %d/%d train loss = %.5f' % (epoch+1, args.max_epochs, epoch_loss/total_train_samples))
      #scheduler.step()
      

      #model_cpy.eval()
      model.eval()
      #mwnet.eval()
      val_total = 0
      val_loss = 0

      class_wise_accuracy = np.zeros(args.class_num)
      validation_loss = nn.CrossEntropyLoss().cuda()
      valloader = dataloaders['val']
      for val_images, val_labels in valloader:
            val_images = val_images.cuda()
            val_labels = val_labels.type(torch.cuda.LongTensor)
            with torch.no_grad():
               out = model(val_images)
               out = classifier(out)
            val_loss += validation_loss(out, val_labels).sum().item()
            #val_loss.backward()
            #mwoptimizer.step()
            _, val_predicted = out.max(1)
          #val_loss += validation_loss(out, val_labels).sum().item()
            val_total += val_labels.size(0)
            #val_correct += val_predicted.eq(val_labels).sum().item()
            for id in range(len(val_predicted)): 
              if val_predicted[id] == val_labels[id]:
                 class_wise_accuracy[int(val_predicted[id])] += 1
     
      class_wise_accuracy = class_wise_accuracy/args.val_samples_per_class
      
      

      val_accuracy = class_wise_accuracy.mean()
      logging.info('Validation: val accuracy = %.4f' % (val_accuracy))
      test_accuracy = test(model, dataloaders, classifier)
      if test_accuracy > best_val_accuracy:
             best_val_accuracy = test_accuracy
             logging.info('Best accuracy = %.4f' % (test_accuracy))
             torch.save(model.state_dict(), os.path.join(args.save_model, 'best_cifar{}_{}_loss_{}_imbalance{}_ce_stage2.pth'.format(args.class_num, args.model, args.loss_type, args.imbalance)))

      



def test(model, dataloaders, classifier):
   
    testloader = dataloaders['test']

    model.eval()
    #mwnet.eval()
    test_total = 0
    test_correct = 0
    class_wise_accuracy = np.zeros(100)
    for test_images, test_labels in testloader:
          test_images = test_images.cuda()
          test_labels = test_labels.type(torch.cuda.LongTensor)
          with torch.no_grad():
               out = model(test_images)
               out = classifier(out)
          _, predicted = out.max(1)
          test_total += test_labels.size(0)
          test_correct += predicted.eq(test_labels).sum().item()
          for id in range(len(predicted)): 
             if predicted[id] == test_labels[id]:
                 class_wise_accuracy[int(predicted[id])] += 1/100
    #with torch.no_grad():
    #   class_weights = mwnet(torch.cuda.FloatTensor(1 - class_wise_accuracy).unsqueeze(0)).squeeze(0)
    #print(class_wise_accuracy)
    #print(class_weights.cpu().numpy())
    #print('Test Accuracy is %.4f'%(test_correct/ test_total))
    return test_correct/test_total 




def main():
    parser = argparse.ArgumentParser(description='Input parameters for CIFAR-LT training')
    parser.add_argument('--class_num', type=int, default=100, help='number of classes (100 for CIFAR100)')
    parser.add_argument('--imbalance', type=int, default=100, help='imbalance ratios in [200, 100, 50, 20, 10, 1(no imbalance)]')

    parser.add_argument('--stage_2_method', type=str, default='LAS', help='[CRT, LWS, LAS]')
    parser.add_argument('--stage1_trained_model', type=str, default='./saved_model/best_cifar100_resnet32_loss_Difficulty_net_imbalance200_stage1.pth')
    parser.add_argument('--save_model', type=str, default='./saved_model/')
    parser.add_argument('--validate_after_every', type=int, default=1, help='validate after every n epochs')
    parser.add_argument('--n_gpus', type=int, default=1, help='number of gpus to use')

    parser.add_argument('--max_epochs', type=int, default=50, help='maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for 1 gpu during training')
    args = parser.parse_args()
    

    pi = 1/args.class_num
    b = -np.log((1-pi)/pi)


    model = build_model(args.class_num)
    model.linear.bias.data.fill_(b)
    model.load_state_dict(torch.load(args.stage1_trained_model),strict=False)
    for parameter in model.parameters():
       parameter.requires_grad = False


    
    print('model prepared')
   

    ###preparation of model done

    ###preparation of data

    ## loading all train images and labels
    
    train_images, train_labels = load_data(args) 
    
    ## loading done
    
    
    
    ## separation of all train data into sub-train and val 

    args.val_samples_per_class = 25
    sub_train_images, sub_train_labels, val_images, val_labels = sep_train_val(train_images, train_labels, args)
    test_images, test_labels, class_names = load_test_data(args)
    
    ## separation done
    
    ## creating imbalance in the dataset
    
    imbalanced_train_images, imbalanced_train_labels = create_imbalance(sub_train_images, sub_train_labels, args)
    
    
    ##imbalance created
    
    imbalanced_train_images = imbalanced_train_images.reshape(-1, 3, 32, 32)  ##reshape to N * C * H * W
    val_images = val_images.reshape(-1, 3, 32, 32)
    test_images = test_images.reshape(-1, 3, 32, 32)

    _, class_wise_freq = np.unique(imbalanced_train_labels, return_counts=True)
    args.freq_ratio = class_wise_freq / class_wise_freq.sum()
    args.class_num_list = class_wise_freq
    
    ###preparation of data done

    ###creation of data loaders for train and val
    sampler_dic = {
                'sampler': source_import('./data/ClassAwareSampler.py').get_sampler(),
                'params': {'num_samples_cls': 4}
            }       #for classawaresampler


    #img_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])
    #img_normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
    #                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    img_normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
            std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0), (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(15),
            AutoAugment(),
            #CIFAR10Policy(),
            Cutout(),
            transforms.ToTensor(),
            #Cutout(n_holes=1, length=16),
            img_normalize,
        ])    
    transform_test = transforms.Compose([transforms.ToTensor(), img_normalize,])
    train_dataset = createCIFAR(imbalanced_train_images, imbalanced_train_labels, transforms=transform_train)
    val_dataset = createCIFAR(val_images, val_labels, transforms=transform_test)
    test_set = createCIFAR(test_images, test_labels, transforms=transform_test)
    sampler = sampler_dic['sampler'](train_dataset, **sampler_dic['params'])
    #sampler = None
    shuffle = False if sampler else True

    trainloader = DataLoader(train_dataset, shuffle=shuffle, sampler=sampler, batch_size=args.batch_size * args.n_gpus, num_workers=8)
    metaloader = DataLoader(val_dataset, shuffle=True, batch_size=100, num_workers=8)
    valloader = DataLoader(val_dataset, shuffle=False, batch_size=256, num_workers=8)
    
    testloader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=8) 
    dataloaders = {'train': trainloader, 'meta': metaloader, 'val': valloader, 'test': testloader}

    ### dataloader creation finished

    train(model, dataloaders, args)
    print('Training Finished')


if __name__ == '__main__':
    main()
          
