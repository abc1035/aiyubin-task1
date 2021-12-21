# -*- coding:GBK -*-
"""
@Time ： 2021/11/12 15:01
@Auth ： aiyubin
@email : aiyubin1999@vip.qq.com
@File ：lapnet_fusion.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)

"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import itertools
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm
import numpy as np
from torch.jit import script
# import geffnet
import argparse
from .lapnet_normal import *
from .resnet_normal import ResNet
from ..builder import BACKBONES
from PIL import Image


@BACKBONES.register_module()
class LapFusion(nn.Module):
    def __init__(self, encoder="resnet", lv6=False, act="RELU", norm="BN", rank=0, max_depth=1, depth=50, plot=False):
        super(LapFusion, self).__init__()
        args = dict(encoder=encoder, lv6=lv6, act=act, norm=norm, rank=rank, max_depth=max_depth)
        args = argparse.Namespace(**args)
        self.encoder = ResNet(depth=50,
                              num_stages=4,
                              out_indices=(0, 1, 2, 3),
                              frozen_stages=1,
                              in_channels=3,
                              norm_cfg=dict(type='BN', requires_grad=True),
                              norm_eval=True,
                              style='pytorch', )
        self.dimList = [64, 256, 512, 1024]
        self.decoder = Lap_decoder_lv5(args, self.dimList)
        self.plot_img = plot
        self.img_metas = None
        self.path = "/home/ayb/origin1/work_dirs/depth_output/"

    def forward(self, x, **kwargs):
        out = self.encoder(x)  # x is a feature list [actrelu, layer1, layer2, layer3, layer4]
        # with dim [64,256,512,1024,2048] and
        # the list will be used in lapnet,and the last four will be used in FPN.
        """for item in out:
            print(item.shape)"""
        depth = self.depth_extractor(x, out[:-1])
        if self.plot_img and False:
            self.plot(depth)
        return tuple(out[1:]), depth

    def depth_extractor(self, x, out_featList):
        rgb_down2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        rgb_down4 = F.interpolate(rgb_down2, scale_factor=0.5, mode='bilinear')
        rgb_down8 = F.interpolate(rgb_down4, scale_factor=0.5, mode='bilinear')
        rgb_down16 = F.interpolate(rgb_down8, scale_factor=0.5, mode='bilinear')
        rgb_down32 = F.interpolate(rgb_down16, scale_factor=0.5, mode='bilinear')
        rgb_up16 = F.interpolate(rgb_down32, rgb_down16.shape[2:], mode='bilinear')
        rgb_up8 = F.interpolate(rgb_down16, rgb_down8.shape[2:], mode='bilinear')
        rgb_up4 = F.interpolate(rgb_down8, rgb_down4.shape[2:], mode='bilinear')
        rgb_up2 = F.interpolate(rgb_down4, rgb_down2.shape[2:], mode='bilinear')
        rgb_up = F.interpolate(rgb_down2, x.shape[2:], mode='bilinear')
        lap1 = x - rgb_up
        lap2 = rgb_down2 - rgb_up2
        lap3 = rgb_down4 - rgb_up4
        lap4 = rgb_down8 - rgb_up8
        lap5 = rgb_down16 - rgb_up16
        rgb_list = [rgb_down32, lap5, lap4, lap3, lap2, lap1]

        d_res_list, depth = self.decoder(out_featList, rgb_list)
        return depth

    def plot(self, depth):
        """save the depth map(array) into a txt file.
        :parameter img_metas: [{'filename':
        '/ayb/UVM_Datasets/voc8/VOCdevkit/VOC2007/JPEGImages/10_263.jpg', 'ori_filename': '10_263.jpg', 'ori_shape':
        (1080, 1920, 3), 'img_shape': (375, 666, 3), 'pad_shape': (384, 672, 3), 'scale_factor': array([0.346875 ,
        0.3472222, 0.346875 , 0.3472222], dtype=float32), 'flip': False, 'flip_direction': None, 'img_norm_cfg': {
        'mean': array([123.675, 116.28 , 103.53 ], dtype=float32), 'std': array([58.395, 57.12 , 57.375],
        dtype=float32), 'to_rgb': True}, 'batch_input_shape': (384, 672)}]
        """
        if self.img_metas is None:
            return
        new_depth = depth.squeeze() * 255
        name = self.img_metas[0]['filename'].split('/')[-1]
        f = open(os.path.join(self.path, name.replace(".jpg", ".txt")), "w")
        f.write(str(new_depth.tolist()))
        f.close()
        print("{} has been writed!".format(os.path.join(self.path, name)))

    def give_img(self, img_metas):
        self.img_metas = img_metas
