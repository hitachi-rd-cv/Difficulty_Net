"""
Copyright (c) Hitachi, Ltd. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the MiSLAS project at https://github.com/dvlab-research/MiSLAS
"""

import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np
import pprint
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F

from datasets.places import Places_LT
from datasets.imagenet import ImageNet_LT

from models import resnet
from models import resnet_places


from utils import config, update_config, create_logger
from utils import AverageMeter, ProgressMeter
from utils import accuracy, calibration

from methods import mixup_data, mixup_criterion
from loss import FocalLoss


class Difficulty_Net(nn.Module):
    def __init__(self, input, hidden1, hidden2, output):
        super(Difficulty_Net, self).__init__()
        self.linear1 = nn.Linear(input, hidden1)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU(inplace=True)
        #self.linear3 = MetaLinear(hidden1, output)
        self.linear3 = nn.Linear(hidden2, output)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        out = self.linear3(x)
        return F.sigmoid(out)



def parse_args():
    parser = argparse.ArgumentParser(description='MiSLAS training (Stage-1)')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(config, args)

    return args


best_acc1 = 0

class BalancedSoftmax(torch.nn.Module):
    def __init__(self, class_freq):
       super(BalancedSoftmax, self).__init__()
       self.class_freq = torch.cuda.FloatTensor(class_freq).unsqueeze(0)
       #self.class_freq = self.class_freq ** 0.25
    def forward(self, logits, target):
       exp_logits = torch.exp(logits) * self.class_freq
       loss = -1 * (torch.log(torch.gather(exp_logits, 1, target.unsqueeze(1))) - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12))
       #logging.info(loss.shape)
       return loss.unsqueeze(1)       


def main():
    args = parse_args()
    logger, model_dir = create_logger(config, args.cfg)
    logger.info('\n' + pprint.pformat(args))
    logger.info('\n' + str(config))

    if config.deterministic:
        seed = 0
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if config.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if config.dist_url == "env://" and config.world_size == -1:
        config.world_size = int(os.environ["WORLD_SIZE"])

    config.distributed = config.world_size > 1 or config.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if config.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        config.world_size = ngpus_per_node * config.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config, logger))
    else:
        # Simply call main_worker function
        main_worker(config.gpu, ngpus_per_node, config, logger, model_dir)


def main_worker(gpu, ngpus_per_node, config, logger, model_dir):
    global best_acc1
    config.gpu = gpu
#     start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    if config.gpu is not None:
        logger.info("Use GPU: {} for training".format(config.gpu))

    if config.distributed:
        if config.dist_url == "env://" and config.rank == -1:
            config.rank = int(os.environ["RANK"])
        if config.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            config.rank = config.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=config.dist_backend, init_method=config.dist_url,
                                world_size=config.world_size, rank=config.rank)

    

    if config.dataset == 'imagenet':
        if 'resnet10' in config.backbone:
           feat_in = 512
        else:
           feat_in = 2048
        model = getattr(resnet, config.backbone)()
        pseudo_model = getattr(resnet, config.backbone)()
        classifier = getattr(resnet, 'Classifier')(feat_in=feat_in, num_classes=config.num_classes)
        pseudo_classifier = getattr(resnet, 'Classifier')(feat_in=feat_in, num_classes=config.num_classes)

    elif config.dataset == 'places':
        model = getattr(resnet_places, config.backbone)(pretrained=True)
        pseudo_model = getattr(resnet_places, config.backbone)(pretrained=False)
        classifier = getattr(resnet_places, 'Classifier')(feat_in=2048, num_classes=config.num_classes)
        pseudo_classifier = getattr(resnet_places, 'Classifier')(feat_in=2048, num_classes=config.num_classes)
        block = getattr(resnet_places, 'Bottleneck')(2048, 512, groups=1, base_width=64, dilation=1, norm_layer=nn.BatchNorm2d)
        pseudo_block = getattr(resnet_places, 'Bottleneck')(2048, 512, groups=1, base_width=64, dilation=1, norm_layer=nn.BatchNorm2d)
    if not torch.cuda.is_available():
        logger.info('using CPU, this will be slow')
    elif config.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if config.gpu is not None:
            torch.cuda.set_device(config.gpu)
            model.cuda(config.gpu)
            classifier.cuda(config.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            config.batch_size = int(config.batch_size / ngpus_per_node)
            config.workers = int((config.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu])
            classifier = torch.nn.parallel.DistributedDataParallel(classifier, device_ids=[config.gpu])
            if config.dataset == 'places':
                block.cuda(config.gpu)
                block = torch.nn.parallel.DistributedDataParallel(block, device_ids=[config.gpu])
        else:
            model.cuda()
            classifier.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            classifier = torch.nn.parallel.DistributedDataParallel(classifier)
            if config.dataset == 'places':
                block.cuda()
                block = torch.nn.parallel.DistributedDataParallel(block)
    elif config.gpu is not None:
        torch.cuda.set_device(config.gpu)
        model = model.cuda(config.gpu)
        classifier = classifier.cuda(config.gpu)
        if config.dataset == 'places':
            block.cuda(config.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()
        pseudo_model = torch.nn.DataParallel(pseudo_model).cuda()
        classifier = torch.nn.DataParallel(classifier).cuda()
        pseudo_classifier = torch.nn.DataParallel(pseudo_classifier).cuda()
        if config.dataset == 'places':
            block = torch.nn.DataParallel(block).cuda()
            pseudo_block = torch.nn.DataParallel(pseudo_block).cuda()
        #else:
        #    pseudo_model = torch.nn.DataParallel(pseudo_model).cuda()

    # optionally resume from a checkpoint
    if config.resume:
        if os.path.isfile(config.resume):
            logger.info("=> loading checkpoint '{}'".format(config.resume))
            if config.gpu is None:
                checkpoint = torch.load(config.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(config.gpu)
                checkpoint = torch.load(config.resume, map_location=loc)
            # config.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if config.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(config.gpu)
            model.load_state_dict(checkpoint['state_dict_model'])
            classifier.load_state_dict(checkpoint['state_dict_classifier'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(config.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(config.resume))

    # Data loading code
    if config.dataset == 'places':
        dataset = Places_LT(config.distributed, root=config.data_path,
                            batch_size=config.batch_size, num_works=config.workers, reuse_val_as_meta=True)

    elif config.dataset == 'imagenet':
        dataset = ImageNet_LT(config.distributed, root=config.data_path,
                              batch_size=config.batch_size, num_works=config.workers, reuse_val_as_meta=True)

    train_loader = dataset.train_instance
    val_loader = dataset.validate
    eval_loader = dataset.eval
    meta_loader = dataset.meta
    cls_num_list = dataset.cls_num_list
    if config.distributed:
        train_sampler = dataset.dist_sampler

    # define loss function (criterion) and optimizer
    if config.bal_softmax:
        criterion = BalancedSoftmax(cls_num_list).cuda(config.gpu)
    else:
        criterion = nn.CrossEntropyLoss(reduction='none').cuda(config.gpu)
    
    #criterion = FocalLoss(gamma=1, size_average=False).cuda(config.gpu)

    if config.dataset == 'places':
        optimizer = torch.optim.SGD([{"params": block.parameters(), "lr": config.lr},
                                    {"params": classifier.parameters(), "lr":config.lr},
                                     { "params": model.parameters(), "lr":0.001}], 
                                    momentum=config.momentum,
                                    weight_decay=config.weight_decay)
        pseudo_optimizer = torch.optim.SGD([{"params": pseudo_block.parameters(), "lr": config.lr},  ##optimizer for intermediate model
                                    {"params": pseudo_classifier.parameters(), "lr": config.lr},     ##
                                    {"params": pseudo_model.parameters(), "lr": 0.001}], 
                                    momentum=config.momentum,
                                    weight_decay=config.weight_decay)

    else:
        optimizer = torch.optim.SGD([{"params": model.parameters()},
                                    {"params": classifier.parameters()}], config.lr,
                                    momentum=config.momentum,
                                    weight_decay=config.weight_decay, nesterov=True)
        pseudo_optimizer = torch.optim.SGD([{"params": pseudo_model.parameters()},
                                    {"params": pseudo_classifier.parameters()}], config.lr,
                                    momentum=config.momentum,
                                    weight_decay=config.weight_decay, nesterov=True)
    diffnet = Difficulty_Net(config.num_classes, config.hidden_layer, config.hidden_layer , config.num_classes).cuda() ##Difficulty-Net
    diffnet_optim = torch.optim.Adam(diffnet.parameters(), lr=config.diff_net_lr, weight_decay=config.diff_net_wd) ##Difficulty-Net optimizer

    class_wise_accuracy = np.zeros(config.num_classes)
    for epoch in range(config.num_epochs):
        if config.distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, config)         ##adjust learning rate
        adjust_learning_rate(pseudo_optimizer, epoch, config)
        
        if config.dataset != 'places':
            block = None
            pseudo_block = None
        # evaluate of validation set
        acc1, class_wise_accuracy = validate(val_loader, model, classifier, criterion, config, logger, block)  ##validation
        # train for one epoch
        train(train_loader, val_loader, meta_loader, model, classifier,  criterion, optimizer, pseudo_optimizer, epoch, config, logger, block, pseudo_model, pseudo_block, pseudo_classifier, diffnet,  diffnet_optim, class_wise_accuracy)
        
        
        # evaluate on test set
        acc1, _ = validate(eval_loader, model, classifier, criterion, config, logger, block)  

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        logger.info('Best Prec@1: %.3f%%\n' % (best_acc1))

        if not config.multiprocessing_distributed or (config.multiprocessing_distributed
                                                      and config.rank % ngpus_per_node == 0):
            if config.dataset == 'places':
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict_model': model.state_dict(),
                    'state_dict_classifier': classifier.state_dict(),
                    'state_dict_block': block.state_dict(),
                    'best_acc1': best_acc1,
                }, is_best, model_dir)

            else:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict_model': model.state_dict(),
                    'state_dict_classifier': classifier.state_dict(),
                    'best_acc1': best_acc1,
                }, is_best, model_dir)

def compute_batch_weights(class_weights, labels):
        y = torch.eye(len(class_weights))
        target_var = labels.cpu()
        labels_one_hot = y[target_var].float().cuda()

        weights = torch.tensor(class_weights).float()
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
        weights = weights.sum(1) 
        return weights




def train(train_loader, val_loader, meta_loader, model, classifier, criterion, optimizer, pseudo_optimizer, epoch, config, logger, block=None, pseudo_model=None, pseudo_block=None, pseudo_classifier=None, diffnet=None,  diffnet_optim=None, accs=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.3f')
    
    top1 = AverageMeter('Acc@1', ':6.3f')
    top5 = AverageMeter('Acc@5', ':6.3f')
    mixup = False
    alpha = 0.2
    mseloss = torch.nn.MSELoss().cuda()
    
    meta_loader_iter = iter(meta_loader)
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    if config.dataset == 'places':
        model.train() ##eval()
        pseudo_model.train()
        block.train()
        pseudo_block.train()
    else:
        model.train()
        pseudo_model.train()
    classifier.train()
    pseudo_classifier.train()

    training_data_num = len(train_loader.dataset)
    end_steps = int(training_data_num / train_loader.batch_size)

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        if i > end_steps:
            break

        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            images = images.cuda(config.gpu, non_blocking=True)
            target = target.cuda(config.gpu, non_blocking=True)
        
        pseudo_classifier.load_state_dict(classifier.state_dict())
        if config.dataset == 'places':
            pseudo_block.load_state_dict(block.state_dict())  ##copying parameters from main model to intermediate model
            pseudo_model.load_state_dict(model.state_dict())  ## ''
        else:
            pseudo_model.load_state_dict(model.state_dict())  ## ''
        diffnet.train()
        class_weights = diffnet(torch.cuda.FloatTensor(accs).unsqueeze(0))
        driver_loss = mseloss(class_weights, torch.cuda.FloatTensor(1 - (accs/(accs.sum() + 1e-12))).unsqueeze(0)) ## driver loss calculation
        class_weights = class_weights.squeeze(0)
        class_weights = class_weights/class_weights.sum() * len(class_weights) ##weight normalization
        batch_weights = compute_batch_weights(class_weights, target)
        if mixup is True:
            images, targets_a, targets_b, indx, lam = mixup_data(images, target, alpha=alpha)
        if config.dataset == 'places':
            feat_a = pseudo_model(images)
            feat = pseudo_block(feat_a)
            output = pseudo_classifier(feat)
        else:
            feat = pseudo_model(images)
            output = pseudo_classifier(feat)
        if mixup is True:
            loss = torch.mean(mixup_criterion(criterion, output, targets_a, targets_b, lam, batch_weights, indx))
        else:
            loss = torch.mean(batch_weights * criterion(output, target))
        pseudo_optimizer.zero_grad()
        loss.backward()
        pseudo_optimizer.step()  ##update intermediate model

        try:
           meta_images, meta_labels = next(meta_loader_iter)
        except StopIteration:
           meta_loader_iter = iter(meta_loader)
           meta_images, meta_labels = next(meta_loader_iter)
        meta_images, meta_labels = meta_images.cuda(), meta_labels.cuda() ##sampling from meta-dataset
        diffnet_optim.zero_grad()
        #feat_a = model(meta_images)
        if config.dataset == 'places':
            feat_a = pseudo_model(meta_images)  ##forward propagate over meta-set mini-batch
            feat = pseudo_block(feat_a)
            output = pseudo_classifier(feat)
        else:
            feat = pseudo_model(meta_images)
            output = pseudo_classifier(feat)
        meta_loss = torch.mean(criterion(output, meta_labels)) + config.lamda * driver_loss ## loss computation for the Difficulty-Net
        meta_loss.backward()
        diffnet_optim.step() ## updating Difficulty-Net
        diffnet.eval()
        with torch.no_grad():
            class_weights = diffnet(torch.cuda.FloatTensor(accs).unsqueeze(0)).squeeze(0)   ##weight computation using updated Difficulty-Net
        class_weights = class_weights/class_weights.sum() * len(class_weights)
        batch_weights = compute_batch_weights(class_weights, target)
        if config.dataset == 'places':
            #with torch.no_grad():
            feat_a = model(images)
            feat = block(feat_a)
            #feat = block(feat_a.detach())
            output = classifier(feat)
        else:
            feat = model(images)
            output = classifier(feat)
        
        if mixup is True:
             loss = torch.mean(mixup_criterion(criterion, output, targets_a, targets_b, lam, batch_weights, indx))
        else:
             loss = torch.mean(batch_weights * criterion(output, target))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() ## updating main model
        
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            progress.display(i, logger)


def validate(val_loader, model, classifier, criterion, config, logger, block=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.3f')
    top1 = AverageMeter('Acc@1', ':6.3f')
    top5 = AverageMeter('Acc@5', ':6.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Eval: ')

    # switch to evaluate mode
    model.eval()
    if config.dataset == 'places':
        block.eval()
    classifier.eval()
    class_num = torch.zeros(config.num_classes).cuda()
    correct = torch.zeros(config.num_classes).cuda()

    confidence = np.array([])
    pred_class = np.array([])
    true_class = np.array([])

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if config.gpu is not None:
                images = images.cuda(config.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(config.gpu, non_blocking=True)

            # compute output
            feat = model(images)
            if config.dataset == 'places':
                feat = block(feat)
            output = classifier(feat)
            loss = torch.mean(criterion(output, target))

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            _, predicted = output.max(1)
            target_one_hot = F.one_hot(target, config.num_classes)
            predict_one_hot = F.one_hot(predicted, config.num_classes)
            class_num = class_num + target_one_hot.sum(dim=0).to(torch.float)
            correct = correct + (target_one_hot + predict_one_hot == 2).sum(dim=0).to(torch.float)

            prob = torch.softmax(output, dim=1)
            confidence_part, pred_class_part = torch.max(prob, dim=1)
            confidence = np.append(confidence, confidence_part.cpu().numpy())
            pred_class = np.append(pred_class, pred_class_part.cpu().numpy())
            true_class = np.append(true_class, target.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.print_freq == 0:
                progress.display(i, logger)

        acc_classes = correct / class_num
        head_acc = acc_classes[config.head_class_idx[0]:config.head_class_idx[1]].mean() * 100

        med_acc = acc_classes[config.med_class_idx[0]:config.med_class_idx[1]].mean() * 100
        tail_acc = acc_classes[config.tail_class_idx[0]:config.tail_class_idx[1]].mean() * 100
        logger.info('* Acc@1 {top1.avg:.3f}% Acc@5 {top5.avg:.3f}% HAcc {head_acc:.3f}% MAcc {med_acc:.3f}% TAcc {tail_acc:.3f}%.'.format(top1=top1, top5=top5, head_acc=head_acc, med_acc=med_acc, tail_acc=tail_acc))

        #cal = calibration(true_class, pred_class, confidence, num_bins=15)
        #logger.info('* ECE   {ece:.3f}%.'.format(ece=cal['expected_calibration_error'] * 100))

    return top1.avg, acc_classes


def save_checkpoint(state, is_best, model_dir):
    filename = model_dir + '/current.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, model_dir + '/model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, config):
    """Sets the learning rate"""
    if config.cos:
        lr_min = 0
        lr_max = config.lr
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(epoch / config.num_epochs * 3.1415926535))
        for param_group in optimizer.param_groups:
           param_group['lr'] = lr
    else:
        epoch = epoch + 1
        #if epoch <= 5:
        #    scale =  epoch / 5
        if epoch == 10:
            scale = 0.1
        elif epoch == 20:
            scale = 0.1
        #elif epoch > 180:
        #    lr = config.lr * 0.01
        #elif epoch > 160:
        #    lr = config.lr * 0.1
        else:
            scale = 1

        for param_group in optimizer.param_groups:
           param_group['lr'] *= scale


if __name__ == '__main__':
    main()
