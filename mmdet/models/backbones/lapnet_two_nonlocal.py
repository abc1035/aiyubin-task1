# -*- coding:GBK -*-
"""
@Time ： 2021/11/12 15:26
@Auth ： aiyubin
@email : aiyubin1999@vip.qq.com
@File ：{name}.py
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
from .resnet_normal2 import ResNet
from ..builder import BACKBONES
from .fuse1_normal import Fuse1
from .fuse2_normal import Fuse2
from .fuse3_normal import Fuse3
from .fuse4_normal import Fuse4
from .fuse6_normal import Fuse6
from .fuck import NLBlockND


@BACKBONES.register_module()
class Lapnet_twoNonlocal(nn.Module):
    def __init__(self, encoder="MobileNetV2", lv6=False, act="RELU", norm="BN", rank=0, max_depth=1):
        super(Lapnet_twoNonlocal, self).__init__()
        args = dict(encoder=encoder, lv6=lv6, act=act, norm=norm, rank=rank, max_depth=max_depth)
        args = argparse.Namespace(**args)
        lv6 = args.lv6
        encoder = args.encoder
        # self.con = torch.cat()
        # self.fuck = torch.zeros()
        if encoder == 'ResNext101':
            self.encoder = deepFeatureExtractor_ResNext101(args, lv6)
        elif encoder == 'VGG19':
            self.encoder = deepFeatureExtractor_VGG19(args, lv6)
        elif encoder == 'DenseNet161':
            self.encoder = deepFeatureExtractor_DenseNet161(args, lv6)
        elif encoder == 'InceptionV3':
            self.encoder = deepFeatureExtractor_InceptionV3(args, lv6)
        elif encoder == 'MobileNetV2':
            self.encoder = deepFeatureExtractor_MobileNetV2(args)
        elif encoder == 'ResNet101':
            self.encoder = deepFeatureExtractor_ResNet101(args, lv6)
        elif 'EfficientNet' in args.encoder:
            self.encoder = deepFeatureExtractor_EfficientNet(args, encoder, lv6)

        if lv6 is True:
            self.decoder = Lap_decoder_lv6(args, self.encoder.dimList)
        else:
            self.decoder = Lap_decoder_lv5(args, self.encoder.dimList)
        # self.encoder2 = ResNet(depth=50, in_channels=4, frozen_stages=1)
        self.rgb_encoder = ResNet(depth=34,
                                  num_stages=4,
                                  out_indices=(0, 1, 2, 3),
                                  frozen_stages=1,
                                  in_channels=3,
                                  norm_cfg=dict(type='BN', requires_grad=True),
                                  norm_eval=True,
                                  style='pytorch', )
        self.d_encoder = ResNet(depth=34,
                                num_stages=4,
                                out_indices=(0, 1, 2, 3),
                                frozen_stages=1,
                                in_channels=1,
                                norm_cfg=dict(type='BN', requires_grad=True),
                                norm_eval=True,
                                style='pytorch', )
        self.plot_img = True
        self.img_metas = None
        self.path = "/home/ayb/origin1/work_dirs/depth_output/"
        # self.fuse = Fuse4()
        # self.non_local1_encoder = NLBlockND(64, dimension=2)
        self.non_local2_encoder = NLBlockND(128, dimension=2)
        self.non_local3_encoder = NLBlockND(256, dimension=2)
        self.non_local4_encoder = NLBlockND(512, dimension=2)

    def forward(self, x, *args, **kwargs):
        """args means that there are some change in forward
        it is tuple whose len is 0 by default.
        if it's len is 1,that mean don't calculate the LapDepth component
        """
        depth = self.depth_extractor(x)
        # depth=torch.cat([depth,depth,depth],dim=1)
        if kwargs.get("iteration", -1) > -2:
            x_depth = self.d_encoder(depth)
        else:
            x_depth = self.d_encoder(depth.detach())
        x_rgb = self.rgb_encoder(x)
        new_x_rgb = []
        # item = self.non_local1_encoder(x_rgb[0], x_depth[0])
        new_x_rgb.append(x_rgb[0])
        item = self.non_local2_encoder(x_rgb[1], x_depth[1])
        new_x_rgb.append(item)
        item = self.non_local3_encoder(x_rgb[2], x_depth[2])
        new_x_rgb.append(item)
        item = self.non_local4_encoder(x_rgb[3], x_depth[3])
        new_x_rgb.append(item)
        if self.plot_img and False:
            self.plot(depth)
        return new_x_rgb, x_depth, depth

    def depth_extractor(self, x):
        out_featList = self.encoder(x)
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
