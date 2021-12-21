# -*- coding:GBK -*-
"""
@Time ： 2021/11/12 15:26
@Auth ： aiyubin
@email : aiyubin1999@vip.qq.com
@File ：{name}.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)

"""

from mmdet.apis import init_detector, inference_detector
import mmcv
from tqdm import tqdm

# Specify the path to model config and checkpoint file
prefix = "/home/ayb/UVM_Datasets/voc8/VOCdevkit/VOC2007/JPEGImages/"
config_file = '/home/ayb/origin1/configs/atss/atss_r34_two_fuse1.py'
checkpoint_file = "/home/ayb/work_dirs/fuse6_after_fpn/epoch_12.pth"
img_dir = '/home/ayb/work_dirs/fuseinfer/'

if __name__ == "__main__":
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    f = open("/home/ayb/work_dirs/fuseinfer/wrong.txt")
    L = []
    while True:
        line = f.readline()
        if line:
            L.append(line.rstrip())
        else:
            break
    f.close()
    for line in tqdm(L):
        img = prefix + line
        result = inference_detector(model, img)
        # print(img_dir + line)
        model.show_result(img, result, out_file=img_dir + line)
