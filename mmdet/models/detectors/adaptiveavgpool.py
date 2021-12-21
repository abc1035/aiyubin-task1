# -*- coding:GBK -*-
"""
@Time ： 2021/11/12 15:26
@Auth ： aiyubin
@email : aiyubin1999@vip.qq.com
@File ：adaptiveavgpool.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)

"""
import torch
import torch.nn as nn
from .convlist import ConvList


class AdaptiveAvgPool(nn.Module):
    def __init__(self, size_list, *args):
        super(AdaptiveAvgPool, self).__init__()
        """Feature fusion.fuse the input of FPN features with depth.motivated by 
        http://mftp.mmcheng.net/Papers/19cvprPoolNet.pdf
        
        Parameter:
           size_list (list) : contains all feature's H&W size in tuple format.
        """

        self.pool_list = []
        #print(size_list)
        for size in size_list:
            self.pool_list.append(nn.AdaptiveAvgPool2d(size))
        if len(args) != 0:  # first pool then conv then add
            self.convlist = args[0]

    def forward(self, x, y, *args):
        """x means the feature tuple before fpn and y means depth which need pooling.y may also be means depth2list"""
        if len(args) == 1 and args[0] == "mutal":
            #print("fuck")
            depth = y
            y_pool_list = []

            for i, pool in enumerate(self.pool_list):
                try:
                    y_pool_list.append(pool(depth))
                except BaseException:
                    continue
            #print(len(y_pool_list))
            #print(len(x))
            feature = []
            for i in range(len(x)):
                feature.append(x[i] * y_pool_list[i])
            return tuple(feature)
        if type(y) == list:
            # depth = y.detach()

            y_pool_list = []
            for i, pool in enumerate(self.pool_list):
                try:
                    y_pool_list.append(pool(y[i]))
                except BaseException:
                    continue
        else:
            # depth = y.detach()
            depth = y

            y_pool_list = []
            for i, pool in enumerate(self.pool_list):
                y_pool_list.append(pool(depth))
        y_pool_list = self.convlist(y_pool_list)  # first pool then conv then add 2021.11.19:19:03
        add_feature = []
        for i, feature in enumerate(x):
            try:
                add_feature.append(torch.add(feature, y_pool_list[i]))
            except BaseException:
                add_feature.append(feature)
        return tuple(add_feature)
