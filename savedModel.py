# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 22:36:21 2023

@author: sharminjouty48
"""

####LateralOptimizationNet

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#from update import BasicUpdateBlock
#from extractor import BasicEncoder
from utils.utils import bilinear_sampler, coords_grid, upflow8

# try:
autocast = torch.cuda.amp.autocast
# except:
#     # dummy autocast for PyTorch < 1.6
#     class autocast:
#         def __init__(self, enabled):
#             pass
#         def __enter__(self):
#             pass
#         def __exit__(self, *args):
#             pass

class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, (7,3), padding=(3,1))
        self.conv2 = nn.Conv2d(hidden_dim, 1, (7,3), padding=(3,1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (7,3), padding=(3,1))
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (7,3), padding=(3,1))
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (7,3), padding=(3,1))

    def forward(self, h, x):
#        print('h',h.shape) 
#        print('x',x.shape) 
        hx = torch.cat([h, x], dim=1)
#        print('hx',hx.shape) 
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h



def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True):
    if isReLU:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True)
        )

class ContextNetwork_PWC(nn.Module):
    def __init__(self, num_ch_in):
        super(ContextNetwork_PWC, self).__init__()

        self.convs = nn.Sequential(
            conv(num_ch_in, 128, 3, 1, 1),
            conv(128, 128, 3, 1, 2),
            conv(128, 128, 3, 1, 4),
            conv(128, 96, 3, 1, 8),
            conv(96, 64, 3, 1, 16),
            conv(64, 32, 3, 1, 1),
            conv(32, 1, isReLU=False)
        )

    def forward(self, x):
        return self.convs(x)
    
class BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(9,3), stride=2, padding=(3,1))
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64,  stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)

        # output convolution
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x
    

class SmallUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=96):
        super(SmallUpdateBlock, self).__init__()
        self.encoder = SmallMotionEncoder(args)
        self.fnet = BasicEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
      #  self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82+64)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=128+128) #inp dim=out dim of fnet
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)
        self.cnet = SmallEncoder(output_dim=hidden_dim, norm_fn='batch', dropout=args.dropout)

        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, (7,3), padding=(3,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))
        
    def forward(self, net, inp,flow, upsample=True):
        inp = torch.cat([inp, flow], dim=1)
#        inp = self.fnet(inp)
#        inp = torch.cat([fmap1, fmap2], dim=1)
 #       print('inp',inp.shape)        
 #       inp = self.fnet(inp)
  #      print('flow',flow.shape) 
#        inp = torch.cat([fmap1, flow], dim=1)
 #       inp=inp.float()
#        inp = torch.cat([inp, flow], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)
        
        # scale mask to balence gradients
        mask = .25 * self.mask(net)
#        return mask, delta_flow
        return net, mask, delta_flow
#        return net, None, delta_flow

   

class RAFT_Lat_opt(nn.Module):
    def __init__(self, args):
        super(RAFT_Lat_opt, self).__init__()
        self.args = args
        self.hidden_dim = hdim = 96
        self.context_dim = cdim = 64
        self.num_ch_in=1

        if 'dropout' not in self.args:
            self.args.dropout = 0


        # feature network, context network, and update block
#        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
        self.cPWCnet = ContextNetwork_PWC(self.num_ch_in)        
        self.fnet = BasicEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
 
        # self.cnet = BasicEncoderSmall(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
        self.cnet = SmallEncoder(output_dim=hdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H, W, device=img.device)
        coords1 = coords_grid(N, H, W, device=img.device)
        # coords0 = coords_grid(N, H//4, W//4, device=img.device)
        # coords1 = coords_grid(N, H//4, W//4, device=img.device)
        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1  
    
    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        #mask = mask.view(N, 1, 9, 4, 4, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        #up_flow = F.unfold(4 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 1, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 1, 8*H, 8*W)
       # return up_flow.reshape(N, 2, 4*H, 4*W)
    def forward(self, image1, image2,iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        # image1 = 2 * (image1 / 255.0) - 1.0
        # image2 = 2 * (image2 / 255.0) - 1.0
        
        image1 = image1.contiguous()
        image2 = image2.contiguous()
        #inp = torch.cat([image1, image2], dim=1)
#        print(image1.shape)
        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            net = self.cnet(image1)
            
            #net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            #net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
#            inp = torch.relu(inp)

        # run the feature network
        # with autocast(enabled=self.args.mixed_precision):
        #     #inp = torch.cat([image1, image2], dim=1)
            
            inp = self.fnet(image1)
        inp = inp.float()
        net = net.float()
#        print('net',net.shape)
#        print('fmap1',fmap1.shape)
        # inp = torch.cat([fmap1, fmap2], dim=1)
        # #        print('inpOfBasicEncoder',inp.shape)        
        # inp = self.fnet(inp)
 #       print('inp',inp.shape)
        coords0, coords1 = self.initialize_flow(image1)
        coords0=coords0[:,0:1,:,:]
        coords1=coords1[:,0:1,:,:]
 
    
        if flow_init is not None:
        #    coords1 = coords1 + flow_init
            flow = flow_init
        flow_predictions = []
        for itr in range(iters):
  #          coords1 = coords1.detach()
            flow = flow.detach()
 #           flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                flow = self.fnet(flow)
                # fmap2 = self.cPWCnet(flow) 
  #               print('fmap2',fmap2.shape)
            #fmap2 = fmap2.float()
            #with autocast(enabled=self.args.mixed_precision):               
                _,up_mask, delta_flow = self.update_block(net, inp, flow)

            # F(t+1) = F(t) + \Delta(t)
            #flow = flow + delta_flow
 #           print('flow',flow.shape)
            # upsample predictions
            delta_flow = self.upsample_flow(delta_flow,up_mask)
 #           delta_flow = upflow8(delta_flow)

#            coords1 = coords1 + delta_flow
     #       print('flow',flow_up.shape)
            flow = flow_init + delta_flow
 #           print('flow',flow.shape)
        #print(up_mask[torch.isnan(up_mask)])
        #print(delta_flow[torch.isnan(delta_flow)])
#            flow_up = self.upsample_flow(flow,up_mask)
 #           coords1 = coords1 + delta_flow
#            flow = flow_init + delta_flow            
            #print('flow',flow_up.shape)
        #flow_up = flow_up.float()

 #           flow_predictions.append(coords1-coords0)
        
        if test_mode:
#            return coords1-coords0
            return flow
            
        return flow


#%%
    
class SmallEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(SmallEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(32)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(32)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 32
        self.layer1 = self._make_layer(32,  stride=1)
        self.layer2 = self._make_layer(64, stride=2)
        self.layer3 = self._make_layer(96, stride=2)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        
        self.conv2 = nn.Conv2d(96, output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = BottleneckBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = BottleneckBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
    
        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x
    
    
class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(BottleneckBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes//4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(planes//4, planes//4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(planes//4, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes//4)
            self.norm2 = nn.BatchNorm2d(planes//4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes//4)
            self.norm2 = nn.InstanceNorm2d(planes//4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm4)


    def forward(self, x):
        y = x
        
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x+y)

  

    
class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=(5,3), padding=(2,1), stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(5,3), padding=(2,1))
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x+y)

class BasicEncoderSmall(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(BasicEncoderSmall, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64,  stride=1)
        self.layer2 = self._make_layer(96, stride=1)
        self.layer3 = self._make_layer(128, stride=1)

        # output convolution
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x
    
class SmallMotionEncoder(nn.Module):
    def __init__(self, args):
        super(SmallMotionEncoder, self).__init__()
       # cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(128, 96, (5,3), padding=(2,1))
        self.convf1 = nn.Conv2d(1, 64, (9,3), padding=(4,1))
        self.convf2 = nn.Conv2d(64, 32, (5,3), padding=(2,1))
        self.conv = nn.Conv2d(128, 80,(5,3), padding=(2,1))

    def forward(self, inp, flow):
     #   print('inp',inp.shape)
     #   print('flow',flow.shape)
        cor = F.relu(self.convc1(inp))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
     #   print('cor',cor.shape)
     #   print('flo',flo.shape)
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)