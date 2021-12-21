# -*- coding:GBK -*-
"""
@Time ： 2021/11/18 18:40
@Auth ： aiyubin
@email : aiyubin1999@vip.qq.com
@File ：convlist.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)

"""
import torch
import torch.nn as nn


class ConvList(nn.Module):
    """when the depth is generated,depth must be processed before add into FPN."""

    def __init__(self, dimList=[256, 512, 1024, 2048]):
        super(ConvList, self).__init__()
        self.convList = nn.ModuleList()
        self.dimList = dimList
        for dim in dimList:
            self.convList.append(nn.Conv2d(in_channels=1, out_channels=dim, kernel_size=3))

    def forward(self, x):
        """x means the depth whose channel==1 and x also can be a list means you process it respectively

        Return:
            depth2list (list) : a list used in add.
        """
        if type(x) == list:
            depth2list = []
            for i, conv in enumerate(self.convList):
                depth2list.append(conv(x[i]))
            return depth2list
        # x = x.detach()
        else:
            depth2list = []
            for i in range(len(self.convList)):
                depth2list.append(self.convList[i](x))

            return depth2list
