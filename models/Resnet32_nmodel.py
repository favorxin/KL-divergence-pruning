import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import math
from res_utils import DownsampleA, DownsampleC, DownsampleD


from torch.autograd import Variable
__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class ResNetBasicblock(nn.Module):
    # expansion is not accurately equals to 4
    expansion = 1

    def __init__(self, inplanes, planes,  index, stride, downsample=None):
        super(ResNetBasicblock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes,kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.index = Variable(index).cuda()

    def forward(self, x):
        residual = x
        out=self.conv1(x)
        out = F.relu(self.bn1(out))
        out=self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        residual.index_add_(1, self.index, out)
        out = F.relu(residual)
        return out


class CifarResNet(nn.Module):

    def __init__(self, layers,index,rate,block=ResNetBasicblock,num_classes=100):
        self.inplanes = rate[0]                        #rate=[16,11,11,11,11,11,11,14,14,14,14,14,14,44,44,44,44,44,44]
        super(CifarResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, rate[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(rate[0])


        index_layer1 = {key: index[key] for key in index.keys() if 'stage_1' in key}
        index_layer2 = {key: index[key] for key in index.keys() if 'stage_2' in key}
        index_layer3 = {key: index[key] for key in index.keys() if 'stage_3' in key}

        self.layer1 = self._make_layer(block, rate[1], 16, index_layer1, layers[0],stride=[1,1,1,1,1])
        self.layer2 = self._make_layer(block,  rate[11], 32, index_layer2,  layers[1], stride=[2,1,1,1,1])
        self.layer3 = self._make_layer(block, rate[21], 64, index_layer3, layers[2], stride=[2,1,1,1,1])


        self.fc = nn.Linear(64, num_classes)
        self.apply(_weights_init)
        '''for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()'''

    def _make_layer(self, block, planes, full_demension, index, blocks, stride ):
        downsample = None
        if stride[0] != 1 or self.inplanes != full_demension:
            downsample=DownsampleA(self.inplanes, planes * block.expansion, stride[0])
        print(index.keys())
        index_block_0_dict = {key: index[key] for key in index.keys() if '0.conv_b' in key}
        index_block_0_value = list(index_block_0_dict.values())[0]
        layers = []
        layers.append(block(self.inplanes, planes, index_block_0_value,stride[0], downsample))
        self.inplanes = full_demension * block.expansion

        for i in range(1, blocks):
            index_block_i_dict = {key: index[key] for key in index.keys() if (str(i) + '.conv_b') in key}
            index_block_i_value = list(index_block_i_dict.values())[0]
            layers.append(block(self.inplanes, planes,index_block_i_value,stride=stride[i]))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out



