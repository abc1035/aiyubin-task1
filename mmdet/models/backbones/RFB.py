# -*- coding:GBK -*-
"""
@Time ： 2021/11/12 15:26
@Auth ： aiyubin
@email : aiyubin1999@vip.qq.com
@File ：{name}.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)

"""
from .resnet_RFB import ResNet,RFB,aggregation
import torch.nn as nn

class RFB(nn.Module):
    def __init__(self):
        super(RFB, self).__init__()
        self.backbone=ResNet()
        self.rfb
