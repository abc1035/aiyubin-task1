# -*- coding:GBK -*-
"""
@Time ： 2021/11/12 15:26
@Auth ： aiyubin
@email : aiyubin1999@vip.qq.com
@File ：{name}.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)

"""
import sys
import os
import numpy as np
from PIL import Image
from ..mmdet.core.visualization import imshow_det_bboxes, imshow_gt_det_bboxes
"""This file aimed to show GT with depth"""
def plot(Path,mode="L"):
    """plot a image from txt"""
    """if os.path.exists(Path.replace(".txt", ".jpg")):
        return"""
    f = open(Path, "r")
    line = f.readline()
    line = eval(line)
    f.close()
    line=np.array(line)
    temp = 255 / (line.max() - line.min())
    line = line - line.min()
    a = line * temp
    print(a.shape)
    # print(a.mean())
    a = a.astype(np.uint8)
    b = Image.fromarray(a,mode=mode)
    b=b.convert(mode='RGB')
    b.save(Path.replace(".txt", ".jpg"))

def plot_a_image(image,annotations,class_names,out_dir):
    file_name=image['file_name']
    image_id=image['id']
    bboxes=[]
    labels=[]
    for ann in annotations:
        if ann['image_id'] == image_id:
            bboxes.append(ann['bbox'])
            labels.append(an['category_id'])
    bboxes=np.array(bboxes)
    labels=np.array(labels)
    out_file=out_dir+file_name
    imshow_gt_det_bboxes(prefix+file_name,bboxes,labels,class_names=class_names,out_file=out_file)


if __name__ =="__main__":
    ann_path = "/home/ayb/UVM_Datasets/voc_test3.json"
    classnames = ['fenda', 'yingyangkuaixian', 'jiaduobao', 'maidong', 'TYCL', 'BSS', 'TYYC', 'LLDS', 'KSFH', 'MZY']
    prefix = '/home/ayb/UVM_Datasets/voc8/VOCdevkit/VOC2007/JPEGImages/'
    save_dir = '/home/ayb/work_dirs/depth/'
    f=open("/home/ayb/UVM_Datasets/voc_test3.json")
    line=f.readline()
    line=eval(line)
    images=line['images']
    annotations=line['annotations']
    image=images[0]
    plot_a_image(image,annotations,classnames,save_dir)


    #plot("/home/ayb/work_dirs/depth.txt")
    """depth_dir='xxx'
    L=os.listdir(depth_dir)
    for item in L:
        jpg_name=item.replace()"""
