# -*- coding:GBK -*-
"""
@Time ： 2021/11/4 16:59
@Auth ： aiyubin
@email : aiyubin1999@vip.qq.com
@File ：depthloss.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)

"""
import torch
import torch.nn as nn
import os
from ..builder import LOSSES
from .utils import weighted_loss
import numpy as np
from .focal_loss import FocalLoss


@LOSSES.register_module()
class DepthLoss(nn.Module):
    def __init__(self, loss_weight):
        super(DepthLoss, self).__init__()
        self.loss_func = FocalLoss()
        self.loss_weight = loss_weight
        self.img_metas = None
        self.path = "/root/origin1/work_dirs/gt/"

    def forward(self, depth, bbox_list, device):
        """
        bbox_list (list): contains N tensors whose shape is (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
        depth (Tensor) : depth Tensor
        """
        # return 0
        loss = 0
        num_gts = len(bbox_list)  # batch_size
        # print(N)
        # print(bbox_list)
        image_size = depth.shape
        overlap = depth[:, :, ] - depth[:, :, ]
        #overlap = overlap.squeeze()
        #print(overlap)
        #print(overlap.shape)
        # overlap = torch.zeros(image_size)
        # print(image_size)
        if torch.cuda.is_available():
            overlap = overlap.type(torch.cuda.LongTensor).cuda(device)
            # overlap=p=overlap.cuda(device)
        # overlap[:, :] = 0
        # overlap = torch.zeros(image_size, device=device)
        # print(num_gts)
        # print(bbox_list)
        for j in range(bbox_list.size(0)):
            # print(bbox_list[j])
            tl_x, tl_y, br_x, br_y = bbox_list[j]
            tl_x, tl_y, br_x, br_y = int(tl_x), int(tl_y), int(br_x), int(br_y)
            # print("{} {} {} {}".format(tl_x, tl_y, br_x, br_y))
            # print(overlap[:, :, tl_y - 1:max(br_y, image_size[1]), tl_x - 1:max(br_x, image_size[0])].shape)
            # print("{} {} {} {}".format(tl_x, tl_y, br_x, br_y))
            # overlap[:, :, tl_x - 1:max(br_x, image_size[0]), tl_y - 1:max(br_y, image_size[1])] = 1
            overlap[:, :, tl_y - 1:max(br_y, image_size[1]), tl_x - 1:max(br_x, image_size[0])] = 1
            overlap[:, :, tl_y - 1:br_y, tl_x - 1:br_x] = 1
        # self.check(overlap)

        # print("############################################################")
        # overlap = overlap.to(depth.device)

        if self.img_metas is not None and False:
            name = self.img_metas[0]['filename'].split('/')[-1]
            f = open(os.path.join(self.path, name.replace(".jpg", ".txt")), "w")
            f.write(str(overlap.tolist()))
            f.close()
            print("{} GT has been writed!".format(os.path.join(self.path, name)))
            self.img_metas = None
        #print(depth.shape)
        # for focal loss
        depth=torch.flatten(depth,start_dim=0)
        overlap=torch.flatten(overlap,start_dim=0)
        # overlap=torch.unsqueeze(dim=0)
        depth1=1-depth
        depth = depth.unsqueeze(dim=0)
        depth1 = depth1.unsqueeze(dim=0)
        depth=torch.cat([depth,depth1],dim=0)
        depth=depth.permute((1,0))
        loss += self.loss_func(depth, overlap)
        '''for i in range(N):  # for every_image
            image_size = depth.shape
            overlap = torch.zeros(image_size)
            num_gts = bbox_list[i].shape[0]
            print(num_gts)
            assert 1 != 1
            for j in range(num_gts):
                tl_x, tl_y, br_x, br_y = bbox_list[i][j]
                overlap[tl_x - 1:max(br_x, image_size[0]), tl_y - 1:max(br_y, image_size[1])] = 1
            loss += self.loss_func(depth, overlap).item()'''
        # print("fuck")
        return loss * self.loss_weight
        pass

    def give_img(self, img_metas):
        self.img_metas = img_metas

    def check(self, overlap):
        temp = overlap.tolist()
        judge = True
        temp = np.array(temp)
        print(np.sum(temp))
        print(temp.size)
        if np.sum(temp) == np.sum(np.where(temp < 1, 1, 1)):
            return True
        else:
            return False
