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
import torch.nn as nn


class Fuse2(nn.Module):
    def __init__(self, channels=[(128, 64), (320, 128), (640, 256), (1280, 512)]):
        super(Fuse2, self).__init__()
        """Fuse1 aims to fuse the feat from RGB branch and D branch from stage 4 out to stage1 out"""
        # channels = [(1024, 512), (1024, 256), (512, 128), (256, 64)]
        self.channels = channels
        self.convs = nn.ModuleList()
        feat_channels = [64, 128, 256, 512]
        self.rgbconvs = nn.ModuleList()
        self.depthconvs = nn.ModuleList()
        self.relu=nn.ReLU()
        for channel in feat_channels:
            self.rgbconvs.append(
                nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1))
            self.depthconvs.append(
                nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1))
        for channel in channels:
            self.convs.append(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, stride=2, padding=1))

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
            x_depth_feat = self.relu(torch.add(x_d_list[i], self.relu(self.depthconvs[i](x_d_list[i]))))
            x_rgb_feat = self.relu(torch.add(x_rgb_list[i], self.relu(self.rgbconvs[i](x_rgb_list[i]))))
            if i == 0:
                x_temp = torch.cat([x_depth_feat, x_rgb_feat], 1)
            else:
                # print(x_temp.shape)
                # print(x_rgb_list[i].shape)
                x_temp = torch.cat([x_depth_feat, x_rgb_feat, x_temp], 1)
            x_temp = self.relu(conv(x_temp))
            feat.append(x_temp)
        return tuple(feat)
