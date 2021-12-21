# -*- coding:GBK -*-
"""
@Time ： 2021/11/12 15:26
@Auth ： aiyubin
@email : aiyubin1999@vip.qq.com
@File ：adaptivemaxpool.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)

"""
import torch
import torch.nn as nn


class AdaptiveMaxPool(nn.Module):
    def __init__(self, size_list):
        super(AdaptiveMaxPool, self).__init__()
        """Feature fusion.fuse the input of FPN features with depth.motivated by 
        http://mftp.mmcheng.net/Papers/19cvprPoolNet.pdf

        Parameter:
           size_list (list) : contains all feature's H&W size in tuple format.
        """

        self.pool_list = []
        for size in size_list:
            self.pool_list.append(nn.AdaptiveMaxPool2d(size))

    def forward(self, x, y):
        """x means the feature tuple before fpn and y means depth which need pooling."""
        depth = y.detach()

        y_pool_list = []
        for pool in self.pool_list:
            y_pool_list.append(pool(depth))

        add_feature = []
        for i, feature in enumerate(x):
            add_feature.append(torch.add(feature, y_pool_list[i]))

        return tuple(add_feature)
