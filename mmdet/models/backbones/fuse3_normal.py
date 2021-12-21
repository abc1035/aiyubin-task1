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


class Fuse3(nn.Module):
    def __init__(self, channels=[64, 128, 256, 512]):
        super(Fuse3, self).__init__()
        """Fuse1 aims to fuse the feat from RGB branch and D branch from stage 4 out to stage1 out"""
        # channels = [(1024, 512), (1024, 256), (512, 128), (256, 64)]
        self.channels = channels
        self.convs = nn.ModuleList()
        self.feat_channels = channels
        self.relu = nn.ReLU()
        for channel in self.feat_channels:
            self.convs.append(
                nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1, padding=0))

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
        for i, conv in enumerate(self.convs):
            x_temp = self.relu(torch.add(x_rgb_list[i], self.relu(self.convs[i](x_d_list[i]))))
            # x_rgb_feat = self.relu(torch.add(x_rgb_list[i], self.relu(self.rgbconvs[i](x_rgb_list[i]))))
            # x_temp = self.relu(conv(x_temp))
            feat.append(x_temp)
        return tuple(feat)
