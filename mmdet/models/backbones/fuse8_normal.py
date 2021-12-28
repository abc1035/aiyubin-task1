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
from torch import nn


class Fuse8(nn.Module):
    def __init__(self, channels=[(64, 128), (128, 256), (256, 512)]):
        super(Fuse8, self).__init__()
        """Fuse1 aims to fuse the feat from RGB branch and D branch from stage 4 out to stage1 out"""
        # channels = [(1024, 512), (1024, 256), (512, 128), (256, 64)]
        self.channels = channels
        # self.convs = nn.ModuleList()
        self.feat_channels = [64,128,256,512]
        self.rgbconvs1 = nn.ModuleList()
        self.depthconvs1 = nn.ModuleList()
        self.rgbconvs2 = nn.ModuleList()
        self.depthconvs2 = nn.ModuleList()
        self.relu = nn.ReLU()
        for channel in self.feat_channels:
            self.rgbconvs1.append(
                nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1))
            self.depthconvs1.append(
                nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1))
            self.rgbconvs2.append(
                nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1))
            self.depthconvs2.append(
                nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1))
        """for channel in channels:
            self.convs.append(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, stride=2, padding=1))"""

    def forward(self, x_d, x_rgb):
        """
        Parameters:
            x_d (tuple) :feat from depth branch with order stage1,2,3,4
            x_rgb : ditto
        """

        x_d_list = list(x_d)
        x_rgb_list = list(x_rgb)
        """x_rgb_list.reverse()
        x_d_list.reverse()"""
        x_temp = None
        feat = []
        for i, conv in enumerate(self.rgbconvs1):
            """print(x_d_list[i].shape)
            print(x_rgb_list[i].shape)"""
            x_depth_temp = self.relu(self.depthconvs1[i](x_d_list[i]))
            x_depth_temp = self.relu(self.depthconvs2[i](x_depth_temp))
            x_rgb_temp = self.relu(self.rgbconvs1[i](x_rgb_list[i]))
            x_rgb_temp = self.relu(self.rgbconvs2[i](x_rgb_temp))
            x_depth_feat = self.relu(torch.add(x_d_list[i], x_depth_temp))
            x_rgb_feat = self.relu(torch.add(x_rgb_list[i], x_rgb_temp))
            """if i == 0:
                x_temp = torch.add(x_depth_feat, x_rgb_feat)
            else:
                # print(x_temp.shape)
                # print(x_rgb_list[i].shape)
                x_temp = self.relu(self.convs[i-1](x_temp))
                x_temp = torch.add(x_depth_feat, torch.add(x_rgb_feat, x_temp))"""
            x_temp = torch.add(x_depth_feat, x_rgb_feat)
            feat.append(x_temp)
        return tuple(feat)
