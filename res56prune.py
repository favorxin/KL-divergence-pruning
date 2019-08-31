import argparse
import numpy as np
import os
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

from models import *
from models_new import *
from scipy.cluster.vq import kmeans,vq,whiten
import scipy.stats
from scipy.spatial import distance
from compute_flops import *

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=56,
                    help='depth of the resnet')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
parser.add_argument('-v', default='A', type=str, 
                    help='version of the model')
parser.add_argument('-type', default='kl', type=str, 
                    help='the type of the pruning method')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)

model = resnet(depth=args.depth, dataset=args.dataset)
#model = resnet56(100)

if args.cuda:
    model.cuda()
if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.model, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

print('Pre-processing Successful!')

# simple test model after Pre-processing prune (simple set BN scales to zeros)
def test(model):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    elif args.dataset == 'cifar100':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    else:
        raise ValueError("No valid dataset is given.")
    model.eval()
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    #print(model)
    end = time.time()
    for i, (input, target) in enumerate(val_loader):

        
        input, target = input.cuda(), target.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        output = model(input_var)
        
        loss = criterion(output, target_var.type(torch.LongTensor).cuda())
        
        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data.cpu(), target.cpu())[0]
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 30== 0:
            print('Test: [{0}/{1}]  '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                  'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))
        #print('==>',top1)

    print('*Prec@1 {top1.avg:.3f}' .format(top1=top1))
    return top1.avg

dataset1 = 'cifar100'

if dataset1 == 'cifar10':
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
elif dataset1 == 'cifar100':
    mean = [x / 255 for x in [129.3, 124.1, 112.4]]
    std = [x / 255 for x in [68.2, 65.4, 70.4]]
else:
    assert False, "Unknow dataset : {}".format(dataset1)

train_transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
     transforms.Normalize(mean, std)])
test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean, std)])

if dataset1 == 'cifar10':
    train_data = datasets.CIFAR10('data/', train=True, transform=train_transform, download=True)
    test_data = datasets.CIFAR10('data/', train=False, transform=test_transform, download=True)
    num_classes = 10
elif dataset1 == 'cifar100':
    train_data = datasets.CIFAR100('data/', train=True, transform=train_transform, download=True)
    test_data = datasets.CIFAR100('data/', train=False, transform=test_transform, download=True)
    num_classes = 100

train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.test_batch_size, shuffle=True,
                                              pin_memory=True)
val_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False,
                                             pin_memory=True)

criterion = nn.CrossEntropyLoss().cuda()
#validate(val_loader,model,criterion)
#acc = test(model)
flops_nums_old = print_model_param_flops(model,32)

skip = {
    'A': [16, 20, 38, 54],
    'B': [16, 18, 20, 34, 38, 54],
}

prun = 0.3
prune_prob = {
    'A': [prun, prun, prun],
    'B': [0.6, 0.3, 0.1],
}

def JS_divergence(p,q):
    M=np.sum([p,q],axis=0)/2
    return 0.5*scipy.stats.entropy(p, M)+0.5*scipy.stats.entropy(q, M)

#def JS_divergence(p,q):
#    e=0.0005
#    list_e=np.zeros(len(p))
#    list_e=[e if i == 0 else i for i in list_e] 
#    #p=np.sum((p,list_e),axis=0)
#    #q=np.sum((q,list_e),axis=0)   
#    M=np.sum([p,q],axis=0)/2
#    return 0.5*scipy.stats.entropy(p, M)+0.5*scipy.stats.entropy(q, M)

prun_type = args.type
layer_id = 1
cfg = []
cfg_mask = []
K=2
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        out_channels = m.weight.data.shape[0]
        if layer_id in skip[args.v]:
            cfg_mask.append(torch.ones(out_channels))
            cfg.append(out_channels)
            layer_id += 1
            continue
        if layer_id % 2 == 0:
            if layer_id <= 18:
                stage = 0
            elif layer_id <= 36:
                stage = 1
            else:
                stage = 2
            prune_prob_stage = prune_prob[args.v][stage]

            if prun_type == 'norm':            
                weight_copy = m.weight.data.abs().clone().cpu().numpy()
                L1_norm = np.sum(weight_copy, axis=(1,2,3))
                num_keep = int(out_channels * (1 - prune_prob_stage))
                arg_max = np.argsort(L1_norm)
                arg_max_rev = arg_max[::-1][:num_keep]            
            elif prun_type == 'gm':
                m_copy = m.weight.data.clone().view(m.weight.data.size()[0],-1)
                num_keep = int(out_channels * (1 - prune_prob_stage))
                similar_matrix = distance.cdist(m_copy, m_copy, 'euclidean')
                similar_sum = np.sum(np.abs(similar_matrix), axis=0)
                arg_max = np.argsort(similar_sum)
                arg_max_rev = arg_max[::-1][:num_keep]            
            elif prun_type == 'kl':    
                KL_matrix = np.zeros([m.weight.data.size()[0],m.weight.data.size()[0]])
                JS_matrix = np.zeros([m.weight.data.size()[0],m.weight.data.size()[0]])
                i=0
                j=0
                num_keep = int(out_channels * (1 - prune_prob_stage))
                m_copy = m.weight.data.clone().view(m.weight.data.size()[0],-1)
                m_abs = np.abs(m_copy.cpu().numpy())
                for j in range(m.weight.data.size()[0]):
                    for i in range(m.weight.data.size()[0]):
                        KL_matrix[j,i] = scipy.stats.entropy(m_abs[j,:],m_abs[i,:])
                        #KL_matrix[j,i] = JS_divergence(m_abs[j,:],m_abs[i,:])
                KL_matrix_sum = np.sum(KL_matrix, axis=0)
                arg_max = np.argsort(KL_matrix_sum)
                arg_max_rev = arg_max[:num_keep]            
            elif prun_type == 'k':            
                m_copy = m.weight.data.clone().view(m.weight.data.size()[0],-1)
                m_np = m_copy.cpu().numpy()
                spot = whiten(m_np)
                center,_ = kmeans(spot,K,20)
                cluster, k_loss = vq(spot,center)                
                num_keep = int(out_channels * (1 - prune_prob_stage))
                arg_max = np.argsort(k_loss)
                arg_max_rev = arg_max[::-1][:num_keep]
            
            mask = torch.zeros(out_channels)
            mask[arg_max_rev.tolist()] = 1
            cfg_mask.append(mask)
            cfg.append(num_keep)
            layer_id += 1
            continue
        layer_id += 1

newmodel = resnet(dataset=args.dataset, depth=args.depth, cfg=cfg)
#newmodel = re_resnet56(100, cfg=cfg)
if args.cuda:
    newmodel.cuda()

start_mask = torch.ones(3)
layer_id_in_cfg = 0
conv_count = 1
for [m0, m1] in zip(model.modules(), newmodel.modules()):
    if isinstance(m0, nn.Conv2d):
        if conv_count == 1:
            m1.weight.data = m0.weight.data.clone()
            conv_count += 1
            continue
        if conv_count % 2 == 0:
            mask = cfg_mask[layer_id_in_cfg]
            idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
            if idx.size == 1:
                idx = np.resize(idx, (1,))
            w = m0.weight.data[idx.tolist(), :, :, :].clone()
            m1.weight.data = w.clone()
            layer_id_in_cfg += 1
            conv_count += 1
            continue
        if conv_count % 2 == 1:
            mask = cfg_mask[layer_id_in_cfg-1]
            idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
            if idx.size == 1:
                idx = np.resize(idx, (1,))
            w = m0.weight.data[:, idx.tolist(), :, :].clone()
            m1.weight.data = w.clone()
            conv_count += 1
            continue
    elif isinstance(m0, nn.BatchNorm2d):
        if conv_count % 2 == 1:
            mask = cfg_mask[layer_id_in_cfg-1]
            idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
            if idx.size == 1:
                idx = np.resize(idx, (1,))
            m1.weight.data = m0.weight.data[idx.tolist()].clone()
            m1.bias.data = m0.bias.data[idx.tolist()].clone()
            m1.running_mean = m0.running_mean[idx.tolist()].clone()
            m1.running_var = m0.running_var[idx.tolist()].clone()
            continue
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()
        m1.running_mean = m0.running_mean.clone()
        m1.running_var = m0.running_var.clone()
    elif isinstance(m0, nn.Linear):
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()

torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'cifar100_resnet110_0.8_kl_1.pth.tar'))

#num_parameters = sum([param.nelement() for param in newmodel.parameters()])
#print("number of parameters: "+str(num_parameters))
#print(newmodel)
model = newmodel
#validate(val_loader,model,criterion)
acc = test(model)
flops_nums_new = print_model_param_flops(newmodel,32)

flops_rate = round(((flops_nums_old-flops_nums_new)/flops_nums_old),4)
print("flops rate:",flops_rate)

#with open(os.path.join(args.save, "cifar100_resnet56_0.9_k.txt"), "w") as fp:
#    fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
#    fp.write("Test accuracy: \n"+str(acc)+"\n")
