import functools
from typing import Any, Sequence, Tuple
import einops
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import transforms
import PIL.Image
import os


def layer_norm_process(feature: torch.Tensor, beta=0., gamma=1., eps=1e-5):
    var_mean = torch.var_mean(feature, dim=-1, unbiased=False)
    # 均值
    mean = var_mean[1]
    # 方差
    var = var_mean[0]

    # layer norm process
    feature = (feature - mean[..., None]) / torch.sqrt(var[..., None] + eps)
    feature = feature * gamma + beta

    return feature

class MlpBlock(nn.Module):
    """A 1-hidden-layer MLP block, applied over the last dimension."""
    def __init__(self, mlp_dim , dropout_rate=0.,use_bias=True):
        super().__init__()
        self.mlp_dim=mlp_dim
        self.dropout_rate=dropout_rate
        self.use_bias=use_bias
        self.fc1 = nn.Linear(self.mlp_dim, self.mlp_dim,bias=self.use_bias)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(self.mlp_dim, self.mlp_dim,bias=self.use_bias)

    def forward(self, x):
        x = x.permute(0,2,3,1)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.permute(0,3,1,2)
        return x

def block_images_einops(x, patch_size):
    """Image to patches."""
    batch, height, width, channels = x.shape
    grid_height = height // patch_size[0]
    grid_width = width // patch_size[1]
    x = einops.rearrange(
        x, "n (gh fh) (gw fw) c -> n (gh gw) (fh fw) c",
        gh=grid_height, gw=grid_width, fh=patch_size[0], fw=patch_size[1])
    return x

def unblock_images_einops(x, grid_size, patch_size):
    """patches to images."""
    x = einops.rearrange(
          x, "n (gh gw) (fh fw) c -> n (gh fh) (gw fw) c",
          gh=grid_size[0], gw=grid_size[1], fh=patch_size[0], fw=patch_size[1])
    return x



class UpSampleRatio(nn.Module):
    """Upsample features given a ratio > 0."""
    def __init__(self, features,b=0, ratio=0., use_bias=True):
        super().__init__()
        self.features = features
        self.ratio = ratio
        self.b = b
        self.bias = use_bias
        self.conv1 = nn.Conv2d(self.features,4*self.features,kernel_size=(1,1),stride=1,bias=self.bias)
        self.conv2 = nn.Conv2d(self.features,2*self.features,kernel_size=(1,1),stride=1,bias=self.bias)
        self.conv3 = nn.Conv2d(self.features,self.features,kernel_size=(1,1),stride=1,bias=self.bias)
        self.conv4 = nn.Conv2d(self.features,self.features//2,kernel_size=(1,1),stride=1,bias=self.bias)
        self.conv5 = nn.Conv2d(self.features,self.features//4,kernel_size=(1,1),stride=1,bias=self.bias)
    def forward(self, x):
        n,c,h,w = x.shape
        # if self.b == 1:
        #     x = x.permute(0,3,1,2)
        x = transforms.Resize(size=( int(h * self.ratio), int(w * self.ratio)))(x)
        # x = x.permute(0,2,3,1)
        if self.b == 0:
            x = self.conv1(x)
        elif self.b==1:
            x = self.conv2(x)
        elif self.b == 2:
            x = self.conv3(x)
        elif self.b == 3:
            x = self.conv4(x)
        elif self.b == 4:
            x = self.conv5(x)
        return x


class CALayer(nn.Module):
    """Squeeze-and-excitation block for channel attention.

    ref: https://arxiv.org/abs/1709.01507
    """
    def __init__(self,a, features, reduction=4, use_bias=True):
        super().__init__()
        self.features = features
        self.reduction = reduction
        self.bias = use_bias
        self.a = a
        self.conv1 = nn.Conv2d(self.features,self.features//self.reduction,kernel_size=(1,1),stride=1,bias=self.bias)
        self.conv3 = nn.Conv2d(self.features*2,self.features//self.reduction,kernel_size=(1,1),stride=1,bias=self.bias)
        self.conv4 = nn.Conv2d(self.features*4,self.features//self.reduction,kernel_size=(1,1),stride=1,bias=self.bias)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(self.features// self.reduction, self.features , kernel_size=(1, 1), stride=1, bias=self.bias)
        self.conv5 = nn.Conv2d(self.features// self.reduction, self.features*2 , kernel_size=(1, 1), stride=1, bias=self.bias)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = torch.mean(x,dim=(2,3),keepdim=True)
        # y = torch.transpose(y,1,3)
        if self.a == 0:
            y = self.conv1(y)
            y = self.relu(y)
            y = self.conv2(y)
            y = self.sigmoid(y)
        elif self.a == 1:
            y = self.conv3(y)
            y = self.relu(y)
            y = self.conv5(y)
            y = self.sigmoid(y)
        else:
            y = self.conv4(y)
        return x * y


class RCAB(nn.Module):#dim
    """Residual channel attention block. Contains LN,Conv,lRelu,Conv,SELayer."""
    def __init__(self,a,features,dim=0 , reduction=4, lrelu_slope=0.2, use_bias=True):
        super().__init__()
        self.features = features
        self.dim = dim
        self.reduction = reduction
        self.lrelu_slope = lrelu_slope
        self.bias = use_bias
        self.layernorm = nn.LayerNorm(dim)
        self.conv1 = nn.Conv2d(self.features,self.features,kernel_size=(3,3),stride=1,bias=self.bias,padding=1)
        self.conv3 = nn.Conv2d(2*self.features,2*self.features,kernel_size=(3,3),stride=1,bias=self.bias,padding=1)
        self.leakly_relu = nn.LeakyReLU(negative_slope=self.lrelu_slope)
        self.conv2 = nn.Conv2d(self.features,self.features,kernel_size=(3,3),stride=1,bias=self.bias,padding=1)
        self.conv4 = nn.Conv2d(2*self.features,2*self.features,kernel_size=(3,3),stride=1,bias=self.bias,padding=1)
        self.calayer = CALayer(features=self.features,reduction=self.reduction,use_bias=self.bias,a=a)
    def forward(self, x):
        shortcut = x
        x = layer_norm_process(x)
        if self.dim == 0:
            x = self.conv1(x)
            x = self.leakly_relu(x)
            x = self.conv2(x)
        else:
            x = self.conv3(x)
            x = self.leakly_relu(x)
            x = self.conv4(x)
        x = self.calayer(x)
        return x + shortcut


class GridGatingUnit(nn.Module):#缺n          dim                                                        n1 = x.shape[-3]    n2,
    """A SpatialGatingUnit as defined in the gMLP paper.

    The 'spatial' dim is defined as the second last.
    If applied on other dims, you should swapaxes first.
    """
    def __init__(self,n1,dim,use_bias=True):
        super().__init__()
        self.bias = use_bias
        self.n1 = n1
        # self.n2 = n2
        self.layernorm = nn.LayerNorm(dim)
        self.fc = nn.Linear(n1,n1,bias=self.bias)
    def forward(self, x):
        c = x.size(-1)
        c = c//2
        u, v = torch.split(x, c, dim=-1)
        v = layer_norm_process(v)
        # v = torch.swapaxes(v, -1, -3)
        # v = einops.rearrange(v,"b c h w -> b h w c")
        v = self.fc(v)
        # v = einops.rearrange(v,"b h w c -> b c h w")
        # v = torch.swapaxes(v, -1, -3)
        return u * (v + 1.)


class GridGmlpLayer(nn.Module):#缺num_channels   n, h, w, num_channels = x.shape         dim   n                                     n2,
    """Grid gMLP layer that performs global mixing of tokens."""
    def __init__(self,n1, dim,grid_size,num_channels, use_bias=True,factor=2,dropout_rate=0.):
        super().__init__()
        self.grid_size = grid_size
        self.num_channels = num_channels
        self.layernorm = nn.LayerNorm(dim)
        self.bias = use_bias
        self.factor = factor
        self.drop = dropout_rate
        self.gelu = nn.GELU()
        self.gridgatingunit = GridGatingUnit(n1,dim=dim,use_bias=self.bias)
        self.dropout = nn.Dropout(self.drop)
        self.fc1 = nn.Linear(num_channels,num_channels*self.factor,bias=self.bias)
        self.fc2 = nn.Linear(num_channels,num_channels,bias=self.bias)
    def forward(self, x):
        # x = x.permute(0,2,3,1)
        n, h, w, num_channels = x.shape
        gh, gw = self.grid_size
        fh, fw = h // gh, w // gw
        x = block_images_einops(x, patch_size=(fh, fw))
        # gMLP1: Global (grid) mixing part, provides global grid communication.
        y = layer_norm_process(x)
        y = self.fc1(y)
        y = self.gelu(y)
        y = self.gridgatingunit(y)
        y = self.fc2(y)
        y = self.dropout(y)
        x = x + y
        x = unblock_images_einops(x, grid_size=(gh, gw), patch_size=(fh, fw))
        return x


class BlockGatingUnit(nn.Module):#缺n    n = x.shape[-2]     dim                                                     ,n2
    """A SpatialGatingUnit as defined in the gMLP paper.

    The 'spatial' dim is defined as the **second last**.
    If applied on other dims, you should swapaxes first.
    """
    def __init__(self,n2,dim, use_bias=True):
        super().__init__()
        self.bias = use_bias
        self.layernorm = nn.LayerNorm(dim)
        self.n2=n2
        self.fc = nn.Linear(n2,n2,bias=self.bias)
    def forward(self, x):
        c = x.size(-1)
        c = c//2
        u, v = torch.split(x, c, dim=-1)
        v = layer_norm_process(v)
        # v = torch.swapaxes(v, -1, -2)
        v = self.fc(v)
        # v = torch.swapaxes(v, -1, -2)
        return u * (v + 1.)

class BlockGmlpLayer(nn.Module): #缺num_channels  n, h, w, num_channels = x.shape        dim   n
    """Block gMLP layer that performs local mixing of tokens."""
    def __init__(self,n2,num_channels, block_size,dim, use_bias=True,factor=2,dropout_rate=0.):
        super().__init__()
        self.block_size = block_size
        self.num_channels = num_channels
        self.bias = use_bias
        self.factor = factor
        self.drop = dropout_rate
        self.layernorm = nn.LayerNorm(dim)
        self.gelu = nn.GELU()
        self.dim=dim
        self.blockgatingunit = BlockGatingUnit(n2=n2,dim=self.dim,use_bias=self.bias)
        self.dropout = nn.Dropout(self.drop)
        self.fc1 = nn.Linear(num_channels,num_channels * self.factor,bias=self.bias)
        self.fc2 = nn.Linear(num_channels,num_channels,bias=self.bias)
    def forward(self, x):
        # x = x.permute(0,2,3,1)
        n, h, w, num_channels = x.shape
        fh, fw = self.block_size
        gh, gw = h // fh, w // fw
        x = block_images_einops(x, patch_size=(fh, fw))
        # MLP2: Local (block) mixing part, provides within-block communication.
        y = layer_norm_process(x)
        y = self.fc1(y)
        y = self.gelu(y)
        y = self.blockgatingunit(y)
        y = self.fc2(y)
        y = self.dropout(y)
        x = x + y
        x = unblock_images_einops(x, grid_size=(gh, gw), patch_size=(fh, fw))
        return x


class ResidualSplitHeadMultiAxisGmlpLayer(nn.Module):#缺 num_channels         n, h, w, num_channels = x.shape        dim
    """The multi-axis gated MLP block."""
    def __init__(self, n1,n2,block_size, grid_size,dim,num_channels, block_gmlp_factor=2,grid_gmlp_factor=2 , input_proj_factor = 2,use_bias=True,dropout_rate=0.):
        super().__init__()
        self.block_size = block_size
        self.grid_size = grid_size
        self.num_channels = num_channels
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.bias = use_bias
        self.drop = dropout_rate
        # self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(num_channels,num_channels * self.input_proj_factor, bias=self.bias)
        self.dim = dim
        self.gelu = nn.GELU()
        self.gridgmlplayer = GridGmlpLayer(n1=n1,dim=3,num_channels=num_channels,grid_size=self.grid_size,factor=self.grid_gmlp_factor,use_bias=self.bias, dropout_rate=self.drop)
        self.blockgmlplayer = BlockGmlpLayer(n2=n2,dim=self.dim,num_channels=num_channels,block_size=self.block_size,factor=self.block_gmlp_factor,  use_bias=self.bias,dropout_rate=self.drop)
        self.fc2 = nn.Linear(num_channels * self.input_proj_factor, num_channels, bias=self.bias)
        self.dropout = nn.Dropout()

    def forward(self, x):
        shortcut = x
        # x = self.norm1(x.permute(0,2,3,1))
        x = layer_norm_process(x.permute(0,2,3,1))
        x = self.fc1(x)
        # x = x.permute(0,3,1,2)
        x = self.gelu(x)
        # x = x.permute(0,2,3,1)
        c = x.size(-1)//2
        u, v = torch.split(x, c, dim=-1)
        # GridGMLPLayer
        u = self.gridgmlplayer(u)
        # BlockGMLPLayer
        v = self.blockgmlplayer(v)
        x = torch.cat([u, v], dim=-1)
        # x = x.permute(0,2,3,1)
        x = self.fc2(x)
        x = x.permute(0,3,1,2)
        x = self.dropout(x)
        # shortcut = torch.transpose(shortcut,1,3)
        x = x + shortcut
        return x


class RDCAB(nn.Module):#缺dim
    """Residual dense channel attention block. Used in Bottlenecks."""
    def __init__(self,a, dim,features, reduction=16, use_bias=True, dropout_rate=0.):
        super().__init__()
        self.features = features
        self.reduction = reduction
        self.bias = use_bias
        self.drop = dropout_rate
        self.norm = nn.LayerNorm(dim)
        self.mlpblock = MlpBlock(mlp_dim=self.features, dropout_rate=self.drop,use_bias=self.bias)
        self.calayer = CALayer(a=a,features=self.features,reduction=self.reduction,use_bias=self.bias)
    def forward(self, x):
        x = x.permute(0,2,3,1)
        y = layer_norm_process(x)
        y = y.permute(0, 3, 1, 2)
        y = self.mlpblock(y)
        y = self.calayer(y)
        x = x.permute(0,3,1,2)
        x = x + y
        return x

class BottleneckBlock(nn.Module):
    """The bottleneck block consisting of multi-axis gMLP block and RDCAB."""
    def __init__(self,a,n1,n2,dim,num_channels,features, block_size, grid_size,num_groups=1,block_gmlp_factor=2,grid_gmlp_factor=2,input_proj_factor=2,channels_reduction=4,use_bias=True, dropout_rate=0.):
        super().__init__()
        self.features = features
        self.block_size = block_size
        self.grid_size = grid_size
        self.num_groups = num_groups
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.channels_reduction = channels_reduction
        self.bias = use_bias
        self.drop = dropout_rate
        self.conv1 = nn.Conv2d(self.features,self.features,kernel_size=(1,1),stride=1)
        self.residualsplitheadmultiaxisgmlpLayer = ResidualSplitHeadMultiAxisGmlpLayer(n1=n1,n2=n2,dim=dim,num_channels=num_channels,grid_size=self.grid_size,block_size=self.block_size,
                grid_gmlp_factor=self.grid_gmlp_factor,block_gmlp_factor=self.block_gmlp_factor,input_proj_factor=self.input_proj_factor,use_bias=self.bias,dropout_rate=self.drop)
        self.rdcab = RDCAB(a=a,dim=dim,features=self.features,reduction=self.channels_reduction,use_bias=self.bias)

    def forward(self, x):
        assert x.ndim == 4  # Input has shape [batch, h, w, c]
        # input projection
        x = self.conv1(x)
        shortcut_long = x
        for i in range(self.num_groups):
            x = self.residualsplitheadmultiaxisgmlpLayer(x)
            # Channel-mixing part, which provides within-patch communication.
            x = self.rdcab(x)
        x = x + shortcut_long
        return x


class UNetEncoderBlock(nn.Module):#缺dim
    """Encoder block in MAXIM."""

    def __init__(self,a,n1,n2,num_channels,dim, features, block_size, grid_size, num_groups=1, lrelu_slope=0.2,block_gmlp_factor=2, grid_gmlp_factor=2,
                input_proj_factor=2, channels_reduction=4, dropout_rate=0., use_bias=True,downsample=True,use_global_mlp=True,use_cross_gating=False,d=0,idx=0,dim_v=64,dim_u=64,f=0,g=0):
        super().__init__()
        self.dim =dim
        self.dim_v = dim_v
        self.dim_u = dim_u
        self.idx = idx
        self.d = d
        self.num_channels = num_channels
        self.features = features
        self.block_size = block_size
        self.grid_size = grid_size
        self.num_groups = num_groups
        self.lrelu_slope = lrelu_slope
        self.block_gmlp_factor = block_gmlp_factor####
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.channels_reduction = channels_reduction
        self.downsample = downsample
        self.use_global_mlp = use_global_mlp
        self.use_cross_gating = use_cross_gating
        self.bias = use_bias
        self.drop = dropout_rate
        self.conv1 = nn.Conv2d(2*self.features,self.features,kernel_size=(1,1),stride=(1,1),bias=self.bias)
        self.conv5 = nn.Conv2d(4*self.features,self.features,kernel_size=(1,1),stride=(1,1),bias=self.bias)
        self.conv3 = nn.Conv2d(self.features,2*self.features,kernel_size=(1,1),stride=(1,1),bias=self.bias)
        self.conv6 = nn.Conv2d(self.features,self.features,kernel_size=(1,1),stride=(1,1),bias=self.bias)
        self.residualsplitheadmultiaxisgmlpLayer = ResidualSplitHeadMultiAxisGmlpLayer(dim=dim,n1=n1,n2=n2,num_channels=num_channels,grid_size=self.grid_size,block_size=self.block_size,
        grid_gmlp_factor=self.grid_gmlp_factor,block_gmlp_factor=self.block_gmlp_factor,input_proj_factor=self.input_proj_factor,
        use_bias=self.bias,dropout_rate=self.drop)
        self.rcab = RCAB(dim=dim,features=self.features,reduction=self.channels_reduction,use_bias=self.bias,a=a)
        self.crossgatingblock = CrossGatingBlock(dim=dim,dim_v=dim_v,dim_u=dim_u,num_channels=num_channels,features=self.features,block_size=self.block_size,grid_size=self.grid_size,
          dropout_rate=self.drop,input_proj_factor=self.input_proj_factor,upsample_y=False,use_bias=self.bias,idx=idx,f=f,g=g)
        self.conv2 = nn.Conv2d(self.features,self.features,kernel_size=(4,4),stride=2,padding=1)
        self.conv7 = nn.Conv2d(self.features,self.features,kernel_size=(4,4),stride=2,padding=1)
        self.conv4 = nn.Conv2d(2*self.features,2*self.features,kernel_size=(4,4),stride=2,padding=1)
        self.conv8 = nn.Conv2d(2*self.features,2*self.features,kernel_size=(4,4),stride=2,padding=1)
    def forward(self, x,skip=None,enc=None,dec=None):

        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            if self.d ==0:
                x = self.conv1(x)
            elif self.d == 1:
                x = self.conv5(x)
            elif self.d==2:
                x = self.conv6(x)
        else:
            x = self.conv3(x)
        shortcut_long = x
        for i in range(self.num_groups):
            if self.use_global_mlp:
                x = self.residualsplitheadmultiaxisgmlpLayer(x)
                x = self.rcab(x)
                x = x + shortcut_long
        if enc is not None and dec is not None:
            assert self.use_cross_gating
            x, _ = self.crossgatingblock(x,enc+dec)
        if self.downsample and self.dim==0 and self.idx==0:
            x_down = self.conv2(x)
            return x_down, x
        elif self.downsample and self.dim == 1 and self.idx==0:
            x_down = self.conv4(x)
            return x_down, x
        elif self.downsample and self.dim==0 and self.idx==1:
            x_down = self.conv7(x.permute(0,3,1,2))
            return x_down, x
        elif self.downsample and self.dim==1 and self.idx==1:
            x_down = self.conv8(x.permute(0,3,1,2))
            return x_down, x
        else:
            return x


class UNetDecoderBlock(nn.Module):
    """Decoder block in MAXIM."""
    def __init__(self,a,d,dim,n1,n2,num_channels, features, block_size, grid_size, num_groups=1, lrelu_slope=0.2, block_gmlp_factor=2,
           grid_gmlp_factor=2,
           input_proj_factor=2, channels_reduction=4, dropout_rate=0., use_bias=True, downsample=True,
           use_global_mlp=True,e = 0):
        super().__init__()
        self.e = e
        self.features = features
        self.block_size = block_size
        self.grid_size = grid_size
        self.num_groups = num_groups
        self.lrelu_slope = lrelu_slope
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.channels_reduction = channels_reduction
        self.downsample = downsample
        self.use_global_mlp = use_global_mlp
        self.bias = use_bias
        self.drop = dropout_rate
        self.conv1 = nn.ConvTranspose2d(self.features,self.features,kernel_size=(2,2),stride=2,bias=self.bias)
        self.conv2 = nn.ConvTranspose2d(self.features,self.features//2,kernel_size=(2,2),stride=2,bias=self.bias)
        self.conv3 = nn.ConvTranspose2d(self.features*2,self.features//2,kernel_size=(2,2),stride=2,bias=self.bias)
        self.conv4 = nn.ConvTranspose2d(self.features*2,self.features,kernel_size=(2,2),stride=2,bias=self.bias)
        self.unetencoderblock = UNetEncoderBlock(a=a,dim=dim,n1=n1,n2=n2,num_channels=num_channels,features=self.features, num_groups=self.num_groups,lrelu_slope=self.lrelu_slope,block_size=self.block_size,
        grid_size=self.grid_size,block_gmlp_factor=self.block_gmlp_factor,grid_gmlp_factor=self.grid_gmlp_factor, channels_reduction=self.channels_reduction,
        use_global_mlp=self.use_global_mlp,dropout_rate=self.drop,downsample=False,use_bias=self.bias,d = d)

    def forward(self, x,bridge=None):
        if self.e==0:
            x = self.conv1(x)
        elif self.e==1:
            x = self.conv2(x)
        elif self.e == 2:
            x = self.conv3(x)
        elif self.e == 3:
            x = self.conv4(x)
        x = self.unetencoderblock(x,skip=bridge)
        return x



class GetSpatialGatingWeights(nn.Module):#缺dim  num_channels  dim_u = u.shape[-3]   dim_v = v.shape[-2]
    """Get gating weights for cross-gating MLP block."""

    def __init__(self, dim,dim_u,dim_v, features,num_channels,block_size,grid_size, input_proj_factor=2, use_bias=True, dropout_rate=0.):
        super().__init__()
        self.dim_u = dim_u
        self.dim_v = dim_v
        self.features = features
        self.num_channels = num_channels
        self.block_size = block_size
        self.grid_size = grid_size
        self.bias = use_bias
        self.drop = dropout_rate
        self.input_proj_factor = input_proj_factor
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(self.num_channels,self.num_channels*self.input_proj_factor,bias=self.bias)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(int(self.num_channels*self.input_proj_factor//2),int(dim_u),bias=self.bias)
        self.fc3 = nn.Linear(int(self.num_channels*self.input_proj_factor//2),int(dim_v),bias=self.bias)
        self.fc4 = nn.Linear(2*self.num_channels,self.num_channels,bias=self.bias)
        self.dropout = nn.Dropout(self.drop)
    def forward(self, x):
        n, h, w, num_channels = x.shape
        # input projection
        x = layer_norm_process(x)
        x = self.fc1(x)
        x = self.gelu(x)
        c = x.size(-1)//2
        u, v = torch.split(x, c, dim=-1)
        # Get grid MLP weights
        gh, gw = self.grid_size
        fh, fw = h // gh, w // gw
        u = block_images_einops(u, patch_size=(fh, fw))
        u = self.fc2(u)
        u = unblock_images_einops(u, grid_size=(gh, gw), patch_size=(fh, fw))
        # Get Block MLP weights
        fh, fw = self.block_size
        gh, gw = h // fh, w // fw
        v = block_images_einops(v, patch_size=(fh, fw))
        v = self.fc3(v)
        v = unblock_images_einops(v, grid_size=(gh, gw), patch_size=(fh, fw))
        x = torch.cat([u, v], dim=-1)
        x = self.fc4(x)
        x = self.dropout(x)
        return x


class CrossGatingBlock(nn.Module):#缺dim     num_channels  n, h, w, num_channels = x.shape
    """Cross-gating MLP block."""
    def __init__(self,dim,dim_v,dim_u,features,block_size,grid_size,num_channels, input_proj_factor=2, use_bias=True,
                 dropout_rate=0.,upsample_y=True,c=0,idx=0,f=0,g=0):
        super().__init__()
        self.features = features
        self.g = g
        self.c=c
        self.f =f
        self.idx=idx
        self.block_size = block_size
        self.num_channels = num_channels
        self.grid_size = grid_size
        self.bias = use_bias
        self.drop = dropout_rate
        self.upsample_y = upsample_y
        self.input_proj_factor = input_proj_factor
        self.conv1 = nn.ConvTranspose2d(self.features,self.features,kernel_size=(2,2),stride=2,bias=self.bias)
        self.conv4 = nn.ConvTranspose2d(2*self.features,self.features,kernel_size=(2,2),stride=2,bias=self.bias)
        self.conv2 = nn.Conv2d(3*self.features,self.features,kernel_size=(1,1),stride=1,bias=self.bias)
        self.conv5 = nn.Conv2d(self.features,self.features,kernel_size=(1,1),stride=1,bias=self.bias)
        self.conv6 = nn.Conv2d(2*self.features,2*self.features,kernel_size=(1,1),stride=1,bias=self.bias)
        self.conv3 = nn.Conv2d(self.features,num_channels,kernel_size=(1,1),stride=1,bias=self.bias)
        self.conv7 = nn.Conv2d(2*self.features,num_channels,kernel_size=(1,1),stride=1,bias=self.bias)
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(self.features,num_channels,bias=self.bias)
        self.fc5 = nn.Linear(2*self.features,num_channels,bias=self.bias)
        self.gelu1 = nn.GELU()
        self.dim = dim
        self.getspatialgatingweights1 = GetSpatialGatingWeights(dim=dim,dim_v=dim_v,dim_u=dim_u,num_channels=num_channels,features=num_channels,block_size=self.block_size,grid_size=self.grid_size,
            dropout_rate=self.drop,use_bias=self.bias)
        self.norm2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(self.features,num_channels,bias=self.bias)
        self.fc6 = nn.Linear(2*self.features,num_channels,bias=self.bias)
        self.gelu2 = nn.GELU()
        self.getspatialgatingweights2 = GetSpatialGatingWeights(dim=self.dim,dim_v=dim_v,dim_u=dim_u,num_channels=num_channels,features=num_channels,block_size=self.block_size,grid_size=self.grid_size,
            dropout_rate=self.drop,use_bias=self.bias)
        self.fc3 = nn.Linear(num_channels,num_channels,bias=self.bias)
        self.dropout1 = nn.Dropout(self.drop)
        self.fc4 = nn.Linear(num_channels,num_channels,bias=self.bias)
        self.dropout2 = nn.Dropout(self.drop)
    def forward(self, x,y):
        # Upscale Y signal, y is the gating signal.
        if self.upsample_y:
            if self.c ==0:
                y = self.conv1(y)
            else:
                y = self.conv4(y)
        if self.idx==0:
            x = self.conv2(x)
        elif self.idx==1 and self.f==0:
            x = self.conv5(x)
        elif self.idx==1 and self.f==1:
            x = self.conv6(x)
        if self.f==0:
            y = self.conv3(y)
        elif self.f==1:
            y = self.conv7(y)
        assert y.shape == x.shape
        y = y.permute(0,2,3,1)
        x = x.permute(0,2,3,1)
        shortcut_x = x
        shortcut_y = y
        # Get gating weights from X
        x = layer_norm_process(x)
        # x = x.permute(0,3,1,2)
        if self.g==0:
            x = self.fc1(x)
        elif self.g==1:
            x = self.fc5(x)
        x = self.gelu1(x)
        gx = self.getspatialgatingweights1(x)
        # Get gating weights from Y
        y = layer_norm_process(y)
        if self.g==0:
            y = self.fc2(y)
        elif self.g==1:
            x = self.fc6(x)
        y = self.gelu2(y)
        gy = self.getspatialgatingweights2(y)
        # Apply cross gating: X = X * GY, Y = Y * GX
        y = y * gx
        y = self.fc3(y)
        y = self.dropout1(y)
        y = y + shortcut_y

        x = x * gy  # gating x using y
        x = self.fc4(x)
        x = self.dropout2(x)
        x = x + y + shortcut_x  # get all aggregated signals
        return x, y


class SAM(nn.Module):
    """Supervised attention module for multi-stage training.

    Introduced by MPRNet [CVPR2021]: https://github.com/swz30/MPRNet
    """
    def __init__(self,features,output_channels=3,use_bias=True):
        super().__init__()
        self.features = features
        self.output_channels = output_channels
        self.bias = use_bias
        self.conv1 = nn.Conv2d(self.features,self.features, kernel_size=(3, 3),bias=self.bias,padding=1)
        self.conv2 = nn.Conv2d(self.features,self.output_channels, kernel_size=(3, 3),bias=self.bias,padding=1)
        self.conv3 = nn.Conv2d(self.features,self.output_channels, kernel_size=(3, 3),bias=self.bias,padding=1)
        self.conv4 = nn.Conv2d(self.output_channels,self.features, kernel_size=(3, 3),bias=self.bias,padding=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x,x_image):
        """Apply the SAM module to the input and features.

        Args:
          x: the output features from UNet decoder with shape (h, w, c)
          x_image: the input image with shape (h, w, 3)
          train: Whether it is training

        Returns:
          A tuple of tensors (x1, image) where (x1) is the sam features used for the
            next stage, and (image) is the output restored image at current stage.
        """
        # Get features
        x1 = self.conv1(x)
        # Output restored image X_s
        if self.output_channels == 3:
            image = self.conv2(x).permute(0,2,3,1) + x_image
        else:
            image = self.conv3(x)
        # Get attention maps for features

        x3 = self.conv4(image.permute(0,3,1,2))
        x2 = self.sigmoid(x3)
        # Get attended feature maps
        x1 = x1 * x2
        # Residual connection
        x1 = x1 + x
        return x1, image

class MAXIM(nn.Module):
    """The MAXIM model function with multi-stage and multi-scale supervision.

    For more model details, please check the CVPR paper:
    MAXIM: MUlti-Axis MLP for Image Processing (https://arxiv.org/abs/2201.02973)

    Attributes:
      features: initial hidden dimension for the input resolution.
      depth: the number of downsampling depth for the model.
      num_stages: how many stages to use. It will also affects the output list.
      num_groups: how many blocks each stage contains.
      use_bias: whether to use bias in all the conv/mlp layers.
      num_supervision_scales: the number of desired supervision scales.
      lrelu_slope: the negative slope parameter in leaky_relu layers.
      use_global_mlp: whether to use the multi-axis gated MLP block (MAB) in each
        layer.
      use_cross_gating: whether to use the cross-gating MLP block (CGB) in the
        skip connections and multi-stage feature fusion layers.
      high_res_stages: how many stages are specificied as high-res stages. The
        rest (depth - high_res_stages) are called low_res_stages.
      block_size_hr: the block_size parameter for high-res stages.
      block_size_lr: the block_size parameter for low-res stages.
      grid_size_hr: the grid_size parameter for high-res stages.
      grid_size_lr: the grid_size parameter for low-res stages.
      num_bottleneck_blocks: how many bottleneck blocks.
      block_gmlp_factor: the input projection factor for block_gMLP layers.
      grid_gmlp_factor: the input projection factor for grid_gMLP layers.
      input_proj_factor: the input projection factor for the MAB block.
      channels_reduction: the channel reduction factor for SE layer.
      num_outputs: the output channels.
      dropout_rate: Dropout rate.

    Returns:
      The output contains a list of arrays consisting of multi-stage multi-scale
      outputs. For example, if num_stages = num_supervision_scales = 3 (the
      model used in the paper), the output specs are: outputs =
      [[output_stage1_scale1, output_stage1_scale2, output_stage1_scale3],
       [output_stage2_scale1, output_stage2_scale2, output_stage2_scale3],
       [output_stage3_scale1, output_stage3_scale2, output_stage3_scale3],]
      The final output can be retrieved by outputs[-1][-1].
    """
    def __init__(self,dim=1,dim2=1,dim4=1,num_channels=64,n1=64,n2=256,features=64,depth=3,num_stages=2,num_groups=1, use_bias=True, num_supervision_scales=int(1), lrelu_slope=0.2,
                 use_global_mlp=True,use_cross_gating=True,high_res_stages=2,block_size_hr=(16,16),block_size_lr=(8,8),
                 grid_size_hr=(16, 16),grid_size_lr=(8, 8),num_bottleneck_blocks=1,
                block_gmlp_factor=2, grid_gmlp_factor=2,input_proj_factor=2, channels_reduction=4, num_outputs=3, dropout_rate=0.):
        super().__init__()
        self.features = features
        self.depth = depth
        self.num_stages = num_stages
        self.num_groups = num_groups
        self.num_supervision_scales = num_supervision_scales
        self.high_res_stages = high_res_stages
        self.block_size_hr = block_size_hr
        self.block_size_lr = block_size_lr
        self.grid_size_hr = grid_size_hr
        self.grid_size_lr = grid_size_lr
        self.num_bottleneck_blocks = num_bottleneck_blocks
        self.num_outputs = num_outputs
        self.lrelu_slope = lrelu_slope
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.channels_reduction = channels_reduction
        self.use_global_mlp = use_global_mlp
        self.use_cross_gating = use_cross_gating
        self.bias = use_bias
        self.drop = dropout_rate
        self.conv1 = nn.Conv2d(3,self.features,kernel_size=(3,3),bias=self.bias,padding=1)
        self.crossgatingblock1 = CrossGatingBlock(dim=dim,dim_v=64,dim_u=64,num_channels=num_channels,features=self.features,
                                                 block_size=self.block_size_hr if 0 < self.high_res_stages else self.block_size_lr,
                                                 grid_size=self.grid_size_hr if 0 < self.high_res_stages else self.block_size_lr,
                                                dropout_rate=self.drop,input_proj_factor=self.input_proj_factor,upsample_y=False,
                                                use_bias=self.bias ,idx=1)
        self.conv2 = nn.Conv2d(self.features,self.features,kernel_size=(1,1),bias=self.bias,padding=1)
        self.conv3 = nn.Conv2d(self.features,self.features*self.features,kernel_size=(3,3),bias=self.bias)
        self.crossgatingblock2 = CrossGatingBlock(dim=dim,dim_v=1,dim_u=1,num_channels=num_channels,features=self.features*self.features,
                                                 block_size=self.block_size_hr if 1 < self.high_res_stages else self.block_size_lr,
                                                 grid_size=self.grid_size_hr if 1 < self.high_res_stages else self.block_size_lr,
                                                dropout_rate=self.drop,input_proj_factor=self.input_proj_factor,upsample_y=False,
                                                use_bias=self.bias)
        self.conv4 = nn.Conv2d(self.features,self.features*self.features,kernel_size=(1,1),bias=self.bias ,padding=1)

        self.unetencoderblock00 = UNetEncoderBlock(a=0,dim=0,n1=64,n2=64,num_channels=num_channels,features=self.features,num_groups=self.num_groups,downsample=True,
            lrelu_slope=self.lrelu_slope,block_size=self.block_size_hr if 0 < self.high_res_stages else self.block_size_lr,
            grid_size=self.grid_size_hr if 0 < self.high_res_stages else self.block_size_lr,block_gmlp_factor=self.block_gmlp_factor,
            grid_gmlp_factor=self.grid_gmlp_factor,input_proj_factor=self.input_proj_factor,channels_reduction=self.channels_reduction,
            use_global_mlp=self.use_global_mlp,dropout_rate=self.drop,use_bias=self.bias,use_cross_gating=False)#i=0 idx_stage=0
        self.unetencoderblock10 = UNetEncoderBlock(a=1,dim=1,num_channels=2*num_channels,n1=128,n2=128,features=self.features,num_groups=self.num_groups,downsample=True,
            lrelu_slope=self.lrelu_slope,block_size=self.block_size_hr if 1 < self.high_res_stages else self.block_size_lr,
            grid_size=self.grid_size_hr if 1 < self.high_res_stages else self.block_size_lr,block_gmlp_factor=self.block_gmlp_factor,
            grid_gmlp_factor=self.grid_gmlp_factor,input_proj_factor=self.input_proj_factor,channels_reduction=self.channels_reduction,
            use_global_mlp=self.use_global_mlp,dropout_rate=self.drop,use_bias=self.bias,use_cross_gating=False)#i=1 idx_stage=0
        self.unetencoderblock20 = UNetEncoderBlock(a=1,dim=dim4,num_channels=4*num_channels,n1=256,n2=256,features=2*self.features,num_groups=self.num_groups,downsample=True,
            lrelu_slope=self.lrelu_slope,block_size=self.block_size_hr if 2 < self.high_res_stages else self.block_size_lr,
            grid_size=self.grid_size_hr if 2 < self.high_res_stages else self.block_size_lr,block_gmlp_factor=self.block_gmlp_factor,
            grid_gmlp_factor=self.grid_gmlp_factor,input_proj_factor=self.input_proj_factor,channels_reduction=self.channels_reduction,
            use_global_mlp=self.use_global_mlp,dropout_rate=self.drop,use_bias=self.bias,use_cross_gating=False)#i=2 idx_stage=0

        self.unetencoderblock01 = UNetEncoderBlock(a=0,dim=0,n1=n1,n2=64,num_channels=num_channels,features=self.features,num_groups=self.num_groups,downsample=True,
            lrelu_slope=self.lrelu_slope,block_size=self.block_size_hr if 0 < self.high_res_stages else self.block_size_lr,
            grid_size=self.grid_size_hr if 0 < self.high_res_stages else self.block_size_lr,block_gmlp_factor=self.block_gmlp_factor,
            grid_gmlp_factor=self.grid_gmlp_factor,input_proj_factor=self.input_proj_factor,channels_reduction=self.channels_reduction,
            use_global_mlp=self.use_global_mlp,dropout_rate=self.drop,use_bias=self.bias,use_cross_gating=True,idx=1)#i=0 idx_stage=1
        self.unetencoderblock11 = UNetEncoderBlock(a=1,dim=1,n1=128,n2=128,num_channels=2*num_channels,features=self.features,num_groups=self.num_groups,downsample=True,
            lrelu_slope=self.lrelu_slope,block_size=self.block_size_hr if 1 < self.high_res_stages else self.block_size_lr,
            grid_size=self.grid_size_hr if 1 < self.high_res_stages else self.block_size_lr,block_gmlp_factor=self.block_gmlp_factor,
            grid_gmlp_factor=self.grid_gmlp_factor,input_proj_factor=self.input_proj_factor,channels_reduction=self.channels_reduction,
            use_global_mlp=self.use_global_mlp,dropout_rate=self.drop,use_bias=self.bias,use_cross_gating=True,idx=1,f=1,g=1,dim_u=128,dim_v=128)#i=1 idx_stage=1
        self.unetencoderblock21 = UNetEncoderBlock(a=1,dim=dim4,n1=256,n2=256,num_channels=4*num_channels,features=2*self.features,num_groups=self.num_groups,downsample=True,
            lrelu_slope=self.lrelu_slope,block_size=self.block_size_hr if 2 < self.high_res_stages else self.block_size_lr,
            grid_size=self.grid_size_hr if 2 < self.high_res_stages else self.block_size_lr,block_gmlp_factor=self.block_gmlp_factor,
            grid_gmlp_factor=self.grid_gmlp_factor,input_proj_factor=self.input_proj_factor,channels_reduction=self.channels_reduction,
            use_global_mlp=self.use_global_mlp,dropout_rate=self.drop,use_bias=self.bias,use_cross_gating=True,idx=1,f=1,g=1,dim_u=256,dim_v=256)#i=2 idx_stage=1

        self.bottleneckblock = BottleneckBlock(a=0,dim=dim,n1=256,n2=256,num_channels=256,block_size=self.block_size_lr,grid_size=self.block_size_lr,features=4 * self.features,
            num_groups=self.num_groups,block_gmlp_factor=self.block_gmlp_factor,grid_gmlp_factor=self.grid_gmlp_factor,input_proj_factor=self.input_proj_factor,
            dropout_rate=self.drop,use_bias=self.bias,channels_reduction=self.channels_reduction)


        self.unsampleratio0 = UpSampleRatio(1*self.features,ratio=2**(-2),use_bias=self.bias,b=0)
        self.unsampleratio1 = UpSampleRatio(2 * self.features,ratio=2**(-1),use_bias=self.bias,b=1)
        self.unsampleratio2 = UpSampleRatio(4 * self.features,ratio=1,use_bias=self.bias,b=2)

        self.unsampleratio3 = UpSampleRatio(1 * self.features,ratio=2**(-1),use_bias=self.bias,b = 1 )
        self.unsampleratio4 = UpSampleRatio(2 * self.features,ratio=2**(0),use_bias=self.bias,b=2)
        self.unsampleratio5 = UpSampleRatio(4 * self.features,ratio=2,use_bias=self.bias,b=3)

        self.unsampleratio6 = UpSampleRatio(1 * self.features,ratio=1,use_bias=self.bias,b=2)
        self.unsampleratio7 = UpSampleRatio(2 * self.features,ratio=2,use_bias=self.bias,b = 3)
        self.unsampleratio8 = UpSampleRatio(4 * self.features,ratio=4,use_bias=self.bias,b=4)


        self.crossgatingblock3 = CrossGatingBlock(dim=dim4,dim_v=256,dim_u=256,num_channels=4*num_channels,features=(2**2) * self.features,
              block_size=self.block_size_hr if 2 < self.high_res_stages else self.block_size_lr,
              grid_size=self.grid_size_hr if 2 < self.high_res_stages else self.block_size_lr,
              input_proj_factor=self.input_proj_factor,
              dropout_rate=self.drop,upsample_y=True,use_bias=self.bias)

        self.conv5 = nn.Conv2d(384,4 * self.features,kernel_size=(1,1),bias=self.bias)
        self.conv6 = nn.Conv2d((2**2) * self.features,(2**2) * self.features,kernel_size=(3,3),bias=self.bias,padding=1)


        self.crossgatingblock4 = CrossGatingBlock(dim=dim2,dim_v=128,dim_u=128,num_channels=2*num_channels,features=(2 ** 1) * self.features,
                                                  block_size=self.block_size_hr if 1 < self.high_res_stages else self.block_size_lr,
                                                  grid_size=self.grid_size_hr if 1 < self.high_res_stages else self.block_size_lr,
                                                  input_proj_factor=self.input_proj_factor,
                                                  dropout_rate=self.drop, upsample_y=True, use_bias=self.bias,c=1)

        self.conv7 = nn.Conv2d((2 ** 1) * self.features, (2 ** 1) * self.features, kernel_size=(1, 1), bias=self.bias,padding=1)
        self.conv8 = nn.Conv2d((2 ** 1) * self.features, (2 ** 1) * self.features, kernel_size=(3, 3), bias=self.bias,padding=1)

        self.crossgatingblock5 = CrossGatingBlock(dim=dim,dim_v=64,dim_u=64,num_channels=num_channels,features=self.features,
                                                  block_size=self.block_size_hr if 0 < self.high_res_stages else self.block_size_lr,
                                                  grid_size=self.grid_size_hr if 0 < self.high_res_stages else self.block_size_lr,
                                                  input_proj_factor=self.input_proj_factor,
                                                  dropout_rate=self.drop, upsample_y=True, use_bias=self.bias,c=1)

        self.conv9 = nn.Conv2d((2 ** 0) * self.features, (2 ** 0) * self.features, kernel_size=(1, 1), bias=self.bias,padding=1)
        self.conv10 = nn.Conv2d((2 ** 0) * self.features, (2 ** 0) * self.features, kernel_size=(3, 3), bias=self.bias,padding=1)


        self.unsampleratio9 = UpSampleRatio(4 * self.features,ratio=2**(0),use_bias=self.bias,b=2)
        self.unsampleratio10 = UpSampleRatio(2 * self.features,ratio=2**(-1),use_bias=self.bias,b=1)
        self.unsampleratio11 = UpSampleRatio(1 * self.features,ratio=2**(-2),use_bias=self.bias)

        self.unsampleratio12 = UpSampleRatio(4 * self.features,ratio=2**(1),use_bias=self.bias,b=3)
        self.unsampleratio13 = UpSampleRatio(2 * self.features,ratio=2**(0),use_bias=self.bias,b=2)
        self.unsampleratio14 = UpSampleRatio(1 * self.features,ratio=2**(-1),use_bias=self.bias,b=1)

        self.unsampleratio15 = UpSampleRatio(4 * self.features,ratio=4,use_bias=self.bias,b=4)
        self.unsampleratio16 = UpSampleRatio(2 * self.features,ratio=2,use_bias=self.bias,b=3)
        self.unsampleratio17 = UpSampleRatio(1 * self.features,ratio=1,use_bias=self.bias,b=2)

        self.unetdecoderblock1 = UNetDecoderBlock(a=0,dim=0,num_channels=4*num_channels,n1=256,n2=n2,features=(2**2) * self.features,num_groups=self.num_groups,
            lrelu_slope=self.lrelu_slope,block_size=self.block_size_hr if 2 < self.high_res_stages else self.block_size_lr,
                                                 grid_size=self.grid_size_hr if 2 < self.high_res_stages else self.block_size_lr,
            block_gmlp_factor=self.block_gmlp_factor,grid_gmlp_factor=self.grid_gmlp_factor,input_proj_factor=self.input_proj_factor,
            channels_reduction=self.channels_reduction,use_global_mlp=self.use_global_mlp,dropout_rate=self.drop,use_bias=self.bias,d =1)

        self.unetdecoderblock2 = UNetDecoderBlock(a=0,dim=0,num_channels=2*num_channels,n1=128,n2=128,features=(2**1) * self.features,num_groups=self.num_groups,
            lrelu_slope=self.lrelu_slope,block_size=self.block_size_hr if 1 < self.high_res_stages else self.block_size_lr,
                                                 grid_size=self.grid_size_hr if 1 < self.high_res_stages else self.block_size_lr,
            block_gmlp_factor=self.block_gmlp_factor,grid_gmlp_factor=self.grid_gmlp_factor,input_proj_factor=self.input_proj_factor,
            channels_reduction=self.channels_reduction,use_global_mlp=self.use_global_mlp,dropout_rate=self.drop,use_bias=self.bias,d = 1,e=3)

        self.unetdecoderblock3 = UNetDecoderBlock(a=0,dim=0,num_channels=num_channels,n1=64,n2=64,features=self.features,num_groups=self.num_groups,
            lrelu_slope=self.lrelu_slope,block_size=self.block_size_hr if 0 < self.high_res_stages else self.block_size_lr,
                                                 grid_size=self.grid_size_hr if 0 < self.high_res_stages else self.block_size_lr,
            block_gmlp_factor=self.block_gmlp_factor,grid_gmlp_factor=self.grid_gmlp_factor,input_proj_factor=self.input_proj_factor,
            channels_reduction=self.channels_reduction,use_global_mlp=self.use_global_mlp,dropout_rate=self.drop,use_bias=self.bias,d =1,e=3)

        self.sam1 = SAM(2 ** (2)*self.features,output_channels=self.num_outputs,use_bias=self.bias)
        self.sam2 = SAM(2 ** (1)*self.features,output_channels=self.num_outputs,use_bias=self.bias)
        self.sam3 = SAM(2 ** (0)*self.features,output_channels=self.num_outputs,use_bias=self.bias)

        self.conv11 = nn.Conv2d((2**(2))*self.features,self.num_outputs,kernel_size=(3,3), bias=self.bias,padding=1)
        self.conv12 = nn.Conv2d((2**(1))*self.features,self.num_outputs,kernel_size=(3,3), bias=self.bias,padding=1)
        self.conv13 = nn.Conv2d((2**(0))*self.features,self.num_outputs,kernel_size=(3,3), bias=self.bias,padding=1)
    def forward(self, x):
        n, h, w, c = x.shape            #bchw
        shortcuts = []
        # x = x.permute(0,3,1,2)
        shortcuts.append(x)
        # Get multi-scale input images
        for i in range(1, self.num_supervision_scales):
            x = transforms.Resize(size=(n, h // (2 ** i), w // (2 ** i), c,),interpolation=transforms.InterpolationMode.NEAREST)(x)
            shortcuts.append(x)

        # store outputs from all stages and all scales
        # Eg, [[(64, 64, 3), (128, 128, 3), (256, 256, 3)],   # Stage-1 outputs
        #      [(64, 64, 3), (128, 128, 3), (256, 256, 3)],]  # Stage-2 outputs
        outputs_all = []
        sam_features, encs_prev, decs_prev = [], [], []
        for idx_stage in range(self.num_stages):
        # Input convolution, get multi-scale input features
            x_scales = []
            for i in range(self.num_supervision_scales):###############################################################################
                if i == 0:
                    x_scale = self.conv1(shortcuts[i].permute(0, 3, 1, 2))
                    # If later stages, fuse input features with SAM features from prev stage
                    if idx_stage > 0:
                        # use larger blocksize at high-res stages
                        if self.use_cross_gating:

                            x_scale, _ = self.crossgatingblock1(x_scale, sam_features.pop())
                        else:
                            cat = torch.cat([x_scale,sam_features.pop()],dim=-1)
                            x_scale = self.conv2(cat)
                    x_scales.append(x_scale)
                else:
                    x_scale = self.conv3(x_scales[i].permute(0, 3, 1, 2))
                    # If later stages, fuse input features with SAM features from prev stage
                    if idx_stage > 0:
                        # use larger blocksize at high-res stages
                        if self.use_cross_gating:
                            x_scale, _ = self.crossgatingblock2(x_scale, sam_features.pop())

                        else:
                            cat = torch.cat([x_scale,sam_features.pop()],dim=-1)
                            x_scale = self.conv4(cat)

                    x_scales.append(x_scale)

            # start encoder blocks
            encs = []
            x = x_scales[0]  # First full-scale input feature
            for i in range(self.depth):  # 0, 1, 2
            # use larger blocksize at high-res stages, vice versa.
                if idx_stage ==0:
                    if i == 0:
                        # Multi-scale input if multi-scale supervision
                        x_scale = x_scales[i] if i < self.num_supervision_scales else None

                        enc_prev = encs_prev.pop() if idx_stage > 0 else None
                        dec_prev = decs_prev.pop() if idx_stage > 0 else None
                        x, bridge = self.unetencoderblock00(x,skip=x_scale,enc=enc_prev,dec=dec_prev)
                        encs.append(bridge)
                    elif i == 1:
                        # Multi-scale input if multi-scale supervision
                        x_scale = x_scales[i] if i < self.num_supervision_scales else None

                        enc_prev = encs_prev.pop() if idx_stage > 0 else None
                        dec_prev = decs_prev.pop() if idx_stage > 0 else None
                        x, bridge = self.unetencoderblock10(x, skip=x_scale, enc=enc_prev, dec=dec_prev)
                        encs.append(bridge)
                    else:
                        # Multi-scale input if multi-scale supervision
                        x_scale = x_scales[i] if i < self.num_supervision_scales else None

                        enc_prev = encs_prev.pop() if idx_stage > 0 else None
                        dec_prev = decs_prev.pop() if idx_stage > 0 else None
                        x, bridge = self.unetencoderblock20(x, skip=x_scale, enc=enc_prev, dec=dec_prev)
                        encs.append(bridge)
                else:
                    if i == 0:
                        # Multi-scale input if multi-scale supervision
                        x_scale = x_scales[i] if i < self.num_supervision_scales else None

                        enc_prev = encs_prev.pop() if idx_stage > 0 else None
                        dec_prev = decs_prev.pop() if idx_stage > 0 else None
                        x, bridge = self.unetencoderblock01(x.permute(0, 3, 1, 2), skip=x_scale.permute(0, 3, 1, 2), enc=enc_prev, dec=dec_prev)
                        encs.append(bridge)
                    elif i == 1:
                        # Multi-scale input if multi-scale supervision
                        x_scale = x_scales[i] if i < self.num_supervision_scales else None

                        enc_prev = encs_prev.pop() if idx_stage > 0 else None
                        dec_prev = decs_prev.pop() if idx_stage > 0 else None
                        x, bridge = self.unetencoderblock11(x, skip=x_scale, enc=enc_prev, dec=dec_prev)
                        encs.append(bridge)
                    else:
                        # Multi-scale input if multi-scale supervision
                        x_scale = x_scales[i] if i < self.num_supervision_scales else None

                        enc_prev = encs_prev.pop() if idx_stage > 0 else None
                        dec_prev = decs_prev.pop() if idx_stage > 0 else None
                        x, bridge = self.unetencoderblock21(x, skip=x_scale, enc=enc_prev, dec=dec_prev)
                        encs.append(bridge)


            # Global MLP bottleneck blocks
            for i in range(self.num_bottleneck_blocks):
                x = self.bottleneckblock(x)

            # cache global feature for cross-gating
            global_feature = x

            # start cross gating. Use multi-scale feature fusion
            skip_features = []
            if idx_stage==1:
                for index in range(len(encs)):
                    encs[index] = encs[index].permute(0,3,1,2)
            for i in reversed(range(self.depth)):  # 2, 1, 0
                if i == 2:
                    # get multi-scale skip signals from cross-gating block
                    signal0 = self.unsampleratio0(encs[0])
                    signal1 = self.unsampleratio1(encs[1])
                    signal2 = self.unsampleratio2(encs[2])

                    signal = torch.cat([signal0,signal1,signal2],dim=1)

                    # Use cross-gating to cross modulate features
                    if self.use_cross_gating:
                        skips, global_feature = self.crossgatingblock3(signal, global_feature)
                        global_feature = global_feature.permute(0,3,1,2)
                    else:
                        skips = self.conv5(signal)
                        skips = self.conv6(skips)
                    skip_features.append(skips)
                elif i == 1:
                    # get multi-scale skip signals from cross-gating block
                    signal0 = self.unsampleratio3(encs[0])
                    signal1 = self.unsampleratio4(encs[1])
                    signal2 = self.unsampleratio5(encs[2])

                    signal = torch.cat([signal0, signal1, signal2], dim=1)
                    # Use cross-gating to cross modulate features
                    if self.use_cross_gating:
                        skips, global_feature = self.crossgatingblock4(signal, global_feature)
                    else:
                        skips = self.conv7(signal)
                        skips = self.conv8(skips)
                    skip_features.append(skips)
                elif i == 0:
                    # get multi-scale skip signals from cross-gating block
                    signal0 = self.unsampleratio6(encs[0])
                    signal1 = self.unsampleratio7(encs[1])
                    signal2 = self.unsampleratio8(encs[2])

                    signal = torch.cat([signal0,signal1,signal2],dim=1)
                    # Use cross-gating to cross modulate features
                    if self.use_cross_gating:
                        skips, global_feature = self.crossgatingblock5(signal, global_feature)
                    else:
                        skips = self.conv9(signal)
                        skips = self.conv10(skips)
                    skip_features.append(skips)

            # for i in skip_features:
            # start decoder. Multi-scale feature fusion of cross-gated features
            outputs, decs, sam_features = [], [], []
            for i in reversed(range(self.depth)):
                if i == 2:
                    # get multi-scale skip signals from cross-gating block
                    signal0 = self.unsampleratio9(skip_features[0].permute(0,3,1,2))
                    signal1 = self.unsampleratio10(skip_features[1].permute(0,3,1,2))
                    signal2 = self.unsampleratio11(skip_features[2].permute(0,3,1,2))

                    signal = torch.cat([signal0, signal1, signal2], dim=1)
                    # Decoder block
                    x = self.unetdecoderblock1(x, bridge=signal)
                    decs.append(x)

                elif i == 1:
                    # get multi-scale skip signals from cross-gating block
                    signal0 = self.unsampleratio12(skip_features[0].permute(0,3,1,2))
                    signal1 = self.unsampleratio13(skip_features[1].permute(0,3,1,2))
                    signal2 = self.unsampleratio14(skip_features[2].permute(0,3,1,2))

                    signal = torch.cat([signal0, signal1, signal2], dim=1)
                    # Decoder block
                    x = self.unetdecoderblock2(x, bridge=signal)
                    decs.append(x)

                elif i == 0:
                    # get multi-scale skip signals from cross-gating block
                    signal0 = self.unsampleratio15(skip_features[0].permute(0,3,1,2))
                    signal1 = self.unsampleratio16(skip_features[1].permute(0,3,1,2))
                    signal2 = self.unsampleratio17(skip_features[2].permute(0,3,1,2))

                    signal = torch.cat([signal0, signal1, signal2], dim=1)

                    # Decoder block
                    x = self.unetdecoderblock3(x, bridge=signal)
                    decs.append(x)
                # output conv, if not final stage, use supervised-attention-block.
                if i < self.num_supervision_scales:
                    if idx_stage < self.num_stages - 1:  # not last stage, apply SAM
                        sam, output = self.sam3(x, shortcuts[i])
                        outputs.append(output)
                        sam_features.append(sam)
                    else:  # Last stage, apply output convolutions
                        output = self.conv13(x)
                        output1 = output.permute(0,2,3,1) + shortcuts[i]
                        outputs.append(output)

            # Cache encoder and decoder features for later-stage's usage
            encs_prev = encs[::-1]
            decs_prev = decs
            # output1 = output1.permute(0,3,1,2)
            # Store outputs
            outputs_all.append(outputs)
        return output1



# maxim = MAXIM()
# input = torch.zeros(size=(2,256,256,3))
# out= maxim(input)
# print(out.shape)