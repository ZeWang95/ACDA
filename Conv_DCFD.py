import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

from torch.nn.parameter import Parameter
import math
import scipy as sp
import scipy.linalg as linalg
import numpy as np
import pdb
from fb import *
from torch.nn.utils import spectral_norm

class Bases_Drop(nn.Module):
    def __init__(self, p):
        super(Bases_Drop, self).__init__()
        self.p = p
    def forward(self, x):
        if self.training:
            assert len(x.shape) == 5
            N, M, L, H, W = x.shape
            mask = torch.ones((N, 1, L, H, W)).float().cuda()*(1-self.p)
            mask = torch.bernoulli(mask) * (1 / (1-self.p))
            x = x * mask
        return x

def bases_list(ks, num_bases):
    len_list = ks // 2
    b_list = []
    for i in range(len_list):
        kernel_size = (i+1)*2+1
        normed_bases, _, _ = calculate_FB_bases(i+1)
        normed_bases = normed_bases.transpose().reshape(-1, kernel_size, kernel_size).astype(np.float32)[:num_bases, ...]

        pad = len_list - (i+1)
        bases = torch.Tensor(normed_bases)
        bases = F.pad(bases, (pad, pad, pad, pad, 0, 0)).view(num_bases, ks*ks)
        b_list.append(bases)
    return torch.cat(b_list, 0)

class Conv_DCFD(nn.Module):
    __constants__ = ['kernel_size', 'stride', 'padding', 'num_bases',
                     'bases_grad', 'mode']
    def __init__(self, in_channels, out_channels, kernel_size, inter_kernel_size=5, stride=1, padding=0, 
        num_bases=6, bias=True,  bases_grad=True, dilation=1, groups=1,
        mode='mode1', bases_drop=None):
        super(Conv_DCFD, self).__init__()
        self.in_channels = in_channels
        self.inter_kernel_size = inter_kernel_size
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_bases = num_bases
        assert mode in ['mode0', 'mode1'], 'Only mode0 and mode1 are available at this moment.'
        self.mode = mode
        self.bases_grad = bases_grad
        self.dilation = dilation
        self.bases_drop = bases_drop
        self.groups = groups

        bases = bases_list(kernel_size, num_bases)
        self.register_buffer('bases', torch.Tensor(bases).float())
        self.tem_size = len(bases)

        bases_size = num_bases * len(bases)

        inter = max(64, bases_size//2)
        self.bases_net = nn.Sequential(
            nn.Conv2d(in_channels, inter, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(inter),
            nn.Tanh(),
            nn.Conv2d(inter, bases_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(bases_size),
            nn.Tanh()
            )

        self.coef = Parameter(torch.Tensor(out_channels, in_channels*num_bases, 1, 1))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.drop = Bases_Drop(p=0.1)


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.coef.size(1))

        nn.init.kaiming_normal_(self.coef, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            self.bias.data.zero_()


    def forward(self, input):
        N, C, H, W = input.shape
        H = H//self.stride
        W = W//self.stride

        M = self.num_bases

        drop_rate = 0.0
        bases = self.bases_net(F.dropout2d(input, p=drop_rate, training=self.training)).view(N, self.num_bases, self.tem_size, H, W) # BxMxMxHxW

        self.bases_coef = bases.cpu().data.numpy()
        bases = torch.einsum('bmkhw, kl->bmlhw', bases, self.bases)
        self.bases_save = bases.cpu().data.numpy()

        x = F.unfold(F.dropout2d(input, p=drop_rate, training=self.training), kernel_size=self.kernel_size, stride=self.stride, padding=self.padding).view(N, self.in_channels, self.kernel_size*self.kernel_size, H, W)
        bases_out = torch.einsum('bmlhw, bclhw-> bcmhw', bases.view(N, self.num_bases, -1, H, W), x).reshape(N, self.in_channels*self.num_bases, H, W)
        bases_out = F.dropout2d(bases_out, p=drop_rate, training=self.training)

        out = F.conv2d(bases_out, self.coef, self.bias)

        return out

    def extra_repr(self):
        return 'kernel_size={kernel_size}, inter_kernel_size={inter_kernel_size}, stride={stride}, padding={padding}, num_bases={num_bases}' \
            ', bases_grad={bases_grad}, mode={mode}, bases_drop={bases_drop}, in_channels={in_channels}, out_channels={out_channels}'.format(**self.__dict__)


if __name__ == '__main__':
    layer = Conv_DCFD(3, 10, kernel_size=3, inter_kernel_size=5, padding=1, stride=2, bias=True).cuda()
    data = torch.randn(10 , 3, 224, 224).cuda()
    print(layer(data).shape)
