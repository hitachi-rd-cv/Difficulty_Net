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
import moco.loader
import moco.builder
from models import *
from losses import *
from data import *
from utils import *
import importlib
import math
import logging
from auto_augment import AutoAugment, Cutout
#from autoaug import CIFAR10Policy, Cutout
from methods import mixup_data, mixup_criterion


logging.basicConfig(level=logging.INFO, filename='cifar-100_imbalance10_no_additional_aug_1.log')

def build_model(class_num):
    model = ResNet32(class_num)

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True

    return model

def adjust_learning_rate(optimizer, epochs):
    # lr = 0.1  * ((0.01 ** int(epochs >= 160)) * (0.01 ** int(epochs >= 210)))
    lr = (0.1 + ((0.1 * epochs/400) * int(epochs <=400)) + (0.1 * int(epochs > 400)))  * ((0.1 ** int(epochs >= 6400)) * (0.1 ** int(epochs >= 9600)))  # For WRN-28-10

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class MWNet(torch.nn.Module):
    def __init__(self):
       super(MWNet, self).__init__()
       self.net = nn.Sequential(nn.Linear(100, 128), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 100), nn.Sigmoid())

    def forward(self, x):
       x = self.net(x)
       return(x)

class BalancedSoftmax(torch.nn.Module):
    def __init__(self, class_freq):
       super(BalancedSoftmax, self).__init__()
       self.class_freq = torch.cuda.FloatTensor(class_freq).unsqueeze(0)
       self.class_freq = self.class_freq# ** 0.5
    def forward(self, logits, target):
       exp_logits = torch.exp(logits) * self.class_freq
       loss = -1 * (torch.log(torch.gather(exp_logits, 1, target.unsqueeze(1))) - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12))
       #logging.info(loss.shape)
       return loss.unsqueeze(1)       
       

def source_import(file_path):
    """This function imports python module directly from source code using importlib"""
    spec = importlib.util.spec_from_file_location('', file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def train(model, model_cpy, mwnet, dataloaders, args):
   
   params_dict = dict(model.named_params(model))
   #print(params_dict.items())
   params = []
   #print(model)
   #model_cpy = model
   for key, value in params_dict.items():
        if key == 'linear.bias':
            params += [{'params':value, 'weight_decay':1e-4}]
        else:
            params += [{'params':value, 'weight_decay':1e-4}]

   ## define optimizer and scheduler
   interim_opt = torch.optim.SGD(model_cpy.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9)
   optimizer = torch.optim.SGD(params, lr=0.1, momentum=0.9)
   mwoptimizer = torch.optim.Adam(mwnet.params(), lr=1e-5, weight_decay=1e-4)

   
   ## create loss
   mseloss = torch.nn.MSELoss().cuda()
   if args.loss_type == 'Difficulty_net':
       loss = torch.nn.CrossEntropyLoss(reduction='none').cuda()
   elif args.loss_type == 'Balanced_Softmax':
       loss = BalancedSoftmax(args.freq_ratio).cuda()
       #bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none').cuda()
       #bce_loss = torch.nn.BCELoss(reduction='none').cuda()
   elif args.loss_type == 'FL':
       loss = FocalLoss(gamma=args.gamma, reduction='none').cuda()
   elif args.loss_type == 'EQL':
       loss = EQLloss(args.freq_ratio, gamma=args.eql_gamma, lamda=args.eql_lambda).cuda()
   elif args.loss_type == 'CDB-CE':
       loss = CDB_loss(class_difficulty = np.ones(args.class_num), tau=args.tau).cuda()
   elif args.loss_type == 'CB':
       loss = CB_Softmax(args.freq_ratio).cuda()
   else :
       sys.exit('Sorry. No such loss function implemented')
   ## create folder for saving model
      
   os.makedirs(args.save_model, exist_ok = True)
   entropy_weights = []

   
   trainloader = dataloaders['train']
   valloader = dataloaders['val']
   metaloader = dataloaders['meta']
   best_val_accuracy = 0
   best_test_accuracy = 0
   best_val_loss = np.inf
   class_weights = torch.ones(args.class_num).cuda()
   class_wise_accuracy = torch.zeros(args.class_num).cuda()

   distance_mat = torch.ones(args.class_num, args.class_num).cuda()

   for epoch in range(args.max_epochs):
      epoch_loss = 0
      total_train_samples = 0
      weight_ent = 0

      adjust_learning_rate(optimizer, epoch)
      #class_weights = mwnet(torch.cuda.FloatTensor(class_wise_accuracy).unsqueeze(0)).squeeze(0)
      meta_loader_iter = iter(metaloader)
      
      for batch_i, (train_images, train_labels) in enumerate(trainloader):
         model.train()
         mwnet.train()
         train_images = Variable(train_images.cuda(), requires_grad=False)
         train_labels = Variable(train_labels.type(torch.cuda.LongTensor), requires_grad=False)
         
         target_var = train_labels.cpu()
         model_cpy = build_model(args.class_num)
         model_cpy.load_state_dict(model.state_dict())
         model_cpy.train()
         class_weights = mwnet(torch.cuda.FloatTensor(class_wise_accuracy).unsqueeze(0))
         diffloss = mseloss(class_weights, torch.cuda.FloatTensor(1 - (class_wise_accuracy/(class_wise_accuracy.sum()+1e-12))).unsqueeze(0))
         class_weights = class_weights.squeeze(0)
         class_weights = class_weights/class_weights.sum() * args.class_num
         #import pdb; pdb.set_trace()
         y = torch.eye(args.class_num)

         labels_one_hot = y[target_var].float().cuda()

         weights = torch.tensor(class_weights).float()
         weights = weights.unsqueeze(0)
         weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
         weights = weights.sum(1) 
         #train_images, targets_a, targets_b, indx, lam = mixup_data(train_images, train_labels, alpha=1.0)
         
         labels_one_hot = F.one_hot(train_labels, args.class_num).type(torch.cuda.FloatTensor)
         #interim_opt.zero_grad()
         out = model_cpy(train_images)
         
         
         #loss = loss / len(train_labels)
         
         #train_loss = torch.mean(mixup_criterion(loss, out, targets_a, targets_b, lam, weights, indx))
         train_loss = loss(out, train_labels) 
         #print(train_loss)   
         #print(train_loss.shape)
         #print(weights.shape)
         train_loss = torch.mean(train_loss * weights)
         model_cpy.zero_grad()
         grads = torch.autograd.grad(train_loss, (model_cpy.params()), create_graph=True)
         #meta_lr = 0.1 * ((0.01 ** int(epoch >= 160)) * (0.01 ** int(epoch >= 210)))   # For ResNet32
         meta_lr = (0.1 + ((0.1 * epoch/400) * int(epoch <=400)) + (0.1 * int(epoch > 400))) * ((0.1 ** int(epoch >= 6400)) * (0.1 ** int(epoch >= 9600)))

         model_cpy.update_params(lr_inner=meta_lr, source_params=grads)
         del grads


      ## meta training

         try:
              meta_images, meta_labels = next(meta_loader_iter)
              #print('tried')
         except StopIteration:
              meta_loader_iter = iter(metaloader)
              meta_images, meta_labels = next(meta_loader_iter)
              #print('restart')

         validation_loss = nn.CrossEntropyLoss().cuda()

         meta_images = meta_images.cuda()
         meta_labels = meta_labels.type(torch.cuda.LongTensor)
         #with torch.no_grad():
         mwoptimizer.zero_grad()
         out = model_cpy(meta_images)
         meta_loss = validation_loss(out, meta_labels) + args.lamda * diffloss
         meta_loss.backward()
         mwoptimizer.step()
            
     
      
      
         #model_cpy.eval()
      #model.train()
         mwnet.eval()
         with torch.no_grad():
              class_weights = mwnet(torch.cuda.FloatTensor(class_wise_accuracy).unsqueeze(0)).squeeze(0)
         # wt = class_weights
         # new_weights = class_weights/class_weights.sum()
         # weight_ent += (-1 * torch.log(new_weights)).mean().item()
         class_weights = class_weights/class_weights.sum() * args.class_num

         target_var = train_labels.cpu()
          
            #import pdb; pdb.set_trace()
         y = torch.eye(args.class_num)

         labels_one_hot = y[target_var].float().cuda()
         weights = torch.tensor(class_weights).float()
         weights = weights.unsqueeze(0)
         weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
         weights = weights.sum(1) 


         out= model(train_images)
         #train_loss = torch.mean(mixup_criterion(loss, out, targets_a, targets_b, lam, weights, indx))
         train_loss = loss(out, train_labels) 
         epoch_loss += torch.sum(train_loss).item()
         total_train_samples += len(train_labels) 
         train_loss = torch.mean(train_loss * weights)
         #train_loss = torch.mean(train_loss)
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
            val_loss += validation_loss(out, val_labels).sum().item()
            _, val_predicted = out.max(1)

            val_total += val_labels.size(0)

            for id in range(len(val_predicted)): 
              if val_predicted[id] == val_labels[id]:
                 class_wise_accuracy[int(val_predicted[id])] += 1
     
      class_wise_accuracy = class_wise_accuracy/args.val_samples_per_class
      
      wt = 1 - class_wise_accuracy
      wt = wt/wt.sum() * args.class_num

      if args.loss_type == 'CDB-CE':
            #cdb_weights = compute_weights(class_weights, tau = args.tau, normalize = args.normalize)
            loss = CDB_loss(class_difficulty=wt, tau=args.tau).cuda()
      val_accuracy = class_wise_accuracy.mean()
      if val_accuracy > best_val_accuracy:
             best_val_accuracy = val_accuracy
             torch.save(model.state_dict(), os.path.join(args.save_model, 'best_cifar{}_{}_loss_{}_imbalance{}_ce_stage1.pth'.format(args.class_num, args.model, args.loss_type, args.imbalance)))
      logging.info('Validation: val accuracy = %.4f' % (val_accuracy))
      test_accuracy = test(model, dataloaders)
      if test_accuracy > best_test_accuracy:
             best_test_accuracy = test_accuracy
             logging.info('Best accuracy = %.4f' % (test_accuracy))
      


def test(model, dataloaders):
   
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
          _, predicted = out.max(1)
          test_total += test_labels.size(0)
          test_correct += predicted.eq(test_labels).sum().item()
          for id in range(len(predicted)): 
             if predicted[id] == test_labels[id]:
                 class_wise_accuracy[int(predicted[id])] += 1/100

    return test_correct/test_total 




def main():
    parser = argparse.ArgumentParser(description='Input parameters for CIFAR-LT training')
    parser.add_argument('--class_num', type=int, default=100, help='number of classes (100 for CIFAR100)')
    parser.add_argument('--imbalance', type=int, default=200, help='imbalance ratios in [200, 100, 50, 20, 10, 1(no imbalance)]')
    parser.add_argument('--model', type=str, default='resnet32', help='[resnet32, vgg16]')
    parser.add_argument('--loss_type', type=str, default='Difficulty_net', help='[Difficulty_net, Balanced_Softmax, EQL (EqualizationLoss), FL (FocalLoss), CDB-CE (Ours)]')
    parser.add_argument('--tau', type=str, default='1', help='[0.5, 1, 1.5, 2, 5, dynamic]')
    parser.add_argument('--lamda', type=float, default=0.3,)
    parser.add_argument('--gamma', type=float, default=1, help='only if you use focal loss')
    parser.add_argument('--eql_gamma', type=float, default=0.9, help='equalization loss gamma')
    parser.add_argument('--eql_lambda', type=float, default=0.005, help='equalization loss lambda')
    parser.add_argument('--save_model', type=str, default='./saved_model/')
    parser.add_argument('--validate_after_every', type=int, default=1, help='validate after every n epochs')
    parser.add_argument('--n_gpus', type=int, default=1, help='number of gpus to use')
    parser.add_argument('--normalize', type=bool, default=True, help='whether to normalise the weights')
    parser.add_argument('--max_epochs', type=int, default=12800, help='maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for 1 gpu during training')
    args = parser.parse_args()
    
    ###prepare ResNet32
    if args.model == 'resnet32':
       model = resnet.resnet32()
       model.linear = nn.Linear(in_features=64, out_features=args.class_num, bias=True)
    elif args.model == 'vgg16':
       model= models.__dict__['vgg19_bn'](num_classes=args.class_num, input_size=32)
       model.classifier = nn.Linear(in_features=512*4, out_features=args.class_num, bias=True)
       #model.features = model.features
       
    else:
       raise NotImplementedError 
    pi = 1/args.class_num
    b = -np.log((1-pi)/pi)
    #model.linear.bias.data.fill_(b)
    #for parameter in model.parameters():
    #   parameter.requires_grad = False
    #for parameter in model.features[43:].parameters():
    #   parameter.requires_grad = True
    #for parameter in model.classifier.parameters():
    #   parameter.requires_grad = True
    model_cpy = resnet.resnet32()
    model_cpy.linear = nn.Linear(in_features=64, out_features=args.class_num, bias=True)
    model_cpy = nn.DataParallel(model_cpy, device_ids=range(args.n_gpus))
    model = nn.DataParallel(model, device_ids=range(args.n_gpus))  #for using multiple gpus
    mwnet = VNet(args.class_num, 128, 128, args.class_num)
    #mwnet = nn.DataParallel(MWNet(), device_ids=range(args.n_gpus))
    #model.load_state_dict(torch.load('./saved_model/best1_cifar100_vgg16_imbalance100.pth'))
    #model.module.classifier.reset_parameters()
    model = build_model(args.class_num)
    #model.linear.bias.data.fill_(b)

    #model = model.cuda()
    mwnet = mwnet.cuda()
    model_cpy = model_cpy.cuda()
    
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
    
    ###preparation of data done

    ###creation of data loaders for train and val
    sampler_dic = {
                'sampler': source_import('./data/ClassAwareSampler.py').get_sampler(),
                'params': {'num_samples_cls': 4}
            }       #for classawaresampler


    #img_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])
    img_normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    img_normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
            std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    #img_normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0), (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            #transforms.RandomGrayscale(p=0.2),
            #transforms.RandomRotation(15),
            AutoAugment(),
            #CIFAR10Policy(),
            Cutout(),
            transforms.ToTensor(),
            #Cutout(n_holes=1, length=16),
            img_normalize,
        ])    
    #augmentation_regular = [
    #    transforms.RandomCrop(32, padding=4),
    #    transforms.RandomHorizontalFlip(),
    #    CIFAR10Policy(),    # add AutoAug
    #    transforms.ToTensor(),
    #    Cutout(n_holes=1, length=16),
    #    transforms.Normalize(
    #        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #    ]
    #augmentation_sim_cifar = [
    #    transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
    #    transforms.RandomHorizontalFlip(),
    #    transforms.RandomApply([
    #        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    #    ], p=0.8),
    #    transforms.RandomGrayscale(p=0.2),
    #   transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
    #    transforms.ToTensor(),
    #    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #]
    #transform_train = transforms.Compose(augmentation_sim_cifar)
    transform_test = transforms.Compose([transforms.ToTensor(), img_normalize,])
    train_dataset = createCIFAR(imbalanced_train_images, imbalanced_train_labels, transforms=transform_train)
    val_dataset = createCIFAR(val_images, val_labels, transforms=transform_test)
    test_set = createCIFAR(test_images, test_labels, transforms=transform_test)
    #sampler = sampler_dic['sampler'](train_dataset, **sampler_dic['params'])
    sampler = None
    shuffle = False if sampler else True

    trainloader = DataLoader(train_dataset, shuffle=shuffle, sampler=sampler, batch_size=args.batch_size * args.n_gpus, num_workers=8)
    metaloader = DataLoader(val_dataset, shuffle=True, batch_size=100, num_workers=8) ## reusing val data as meta data
    valloader = DataLoader(val_dataset, shuffle=False, batch_size=256, num_workers=8)
    
    testloader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=8) 
    dataloaders = {'train': trainloader, 'meta': metaloader, 'val': valloader, 'test': testloader}

    ### dataloader creation finished

    train(model, model_cpy, mwnet, dataloaders, args)
    print('Training Finished')


if __name__ == '__main__':
    main()
          
