# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .single_stage import SingleStageDetector
from .adaptiveavgpool import AdaptiveAvgPool
from .adaptivemaxpool import AdaptiveMaxPool
from .convlist import ConvList
from ..backbones.fuse5_normal import Fuse5
from ..backbones.fuse6_normal import Fuse6
from ..backbones.fuse7_normal import Fuse7
import torch
import numpy as np


@DETECTORS.register_module()
class ATSS(SingleStageDetector):
    """Implementation of `ATSS <https://arxiv.org/abs/1912.02424>`_."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(ATSS, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)

        self.convlist = ConvList()
        self.iteration = 0
        # self.fuse = Fuse7()

    def extract_feat(self, img, *args, **kwargs):
        if len(args) == 2:
            self.backbone.give_img(args[1])
        x,y=self.backbone(img)
        if self.with_neck:
            x=self.neck(x)
        return x,y

    def extract_feat1(self, img):
        x, y = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def updatedic(self, dic, info):
        """update the key of dic
        Parameters:
            info(str) :key+info
        """

        new_dic = {}
        for key in dic:
            new_dic[key + info] = dic[key]
        return new_dic

    def extract_feat_fuse_after(self, img, *args, **kwargs):
        """fuse the feature after FPN"""
        temp = self.backbone(img, self.iteration)
        assert len(temp) == 3
        x_rgb, x_depth, depth = temp
        if self.with_neck:
            x_rgb = self.neck(x_rgb)
            x_depth = self.neck1(x_depth)
        """for item in x_rgb:
            print(item.shape)"""
        x = self.fuse(x_depth, x_rgb)
        """for item in x:
            print(item.shape)"""
        """for i, item in enumerate(x_depth):
            item = np.array((item.sum(dim=1) / item.shape[1]).squeeze().cpu()).tolist()
            f = open("/home/ayb/work_dirs/depth{}.txt".format(i), "w")
            f.write(str(item))
            f.close()
        assert 1 == 2"""
        if len(args) == 0:
            return x, x_rgb, x_depth, depth
        else:
            """x = x[0]
            print(x)
            print(x.shape)
            x = np.array((x.sum(dim=1) / x.shape[1]).squeeze().cpu()).tolist()
            f = open("/home/ayb/work_dirs/x.txt", "w")
            f.write(str(x))
            f.close()
            x_rgb = x_rgb[0]
            x_rgb = np.array((x_rgb.sum(dim=1) / x_rgb.shape[1]).squeeze().cpu()).tolist()
            f1 = open("/home/ayb/work_dirs/x_rgb.txt", "w")
            f1.write(str(x_rgb))
            f1.close()
            depth = np.array((depth.sum(dim=1) / depth.shape[1]).squeeze().cpu()).tolist()
            f2 = open("/home/ayb/work_dirs/depth.txt", "w")
            f2.write(str(depth))
            f2.close()
            assert 1 == 2"""
            branch = kwargs.get("branch", "main")
            if branch == "main":
                return x
            elif branch == "rgb":
                return x_rgb
