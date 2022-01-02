# -*- coding:GBK -*-
"""
@Time ： 2021/11/12 15:26
@Auth ： aiyubin
@email : aiyubin1999@vip.qq.com
@File ：{name}.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)

"""
"""this file is aimed to plot predict bbox and GT bbox"""
from mmdet.apis import init_detector, inference_detector
import mmcv
from tqdm import tqdm
import gc
import os
import psutil

ann_path = "/home/ayb/UVM_Datasets/voc_test_for_depth.json"
classnames = ['fenda', 'yingyangkuaixian', 'jiaduobao', 'maidong', 'TYCL', 'BSS', 'TYYC', 'LLDS', 'KSFH', 'MZY']
config_file = '/home/ayb/origin1/configs/atss/atss_lcx1.py'
checkpoint_file = "/home/ayb/epoch_10.pth"
#prefix = '/home/ayb/UVM_Datasets/voc8/VOCdevkit/VOC2007/JPEGImages/'
prefix = '/home/ayb/work_dirs/depth/fuse/'
# save_dir = '/home/ayb/work_dirs/nonlocalerror/'
save_dir='/home/ayb/work_dirs/depth/compare/'
error_path = "/home/ayb/work_dirs/depth/wrong1.txt" # has been modified
model = init_detector(config_file, checkpoint_file, device='cuda:0')

f = open(ann_path, "r")
line = f.readline()
line = eval(line)
f.close()
images = line['images']
annotations = line['annotations']
f = open(error_path, "r")
errors = []
while True:
    line = f.readline()
    if line:
        errors.append(line.rstrip())
    else:
        break
f.close()
# exist = os.listdir(save_dir)
exist=[]


def get_bbox(id):
    bbox_list = [x['bbox'] if x['image_id'] == id else None for x in annotations]
    bbox_list = list(filter(lambda x: x != None, bbox_list))
    for i, item in enumerate(bbox_list):
        item[2] = item[0] + item[2]
        item[3] = item[1] + item[3]
        bbox_list[i] = item
    cat_list = [x['category_id'] if x['image_id'] == id else None for x in annotations]
    cat_list = list(filter(lambda x: x != None, cat_list))
    new_cat_list = [x - 1 for x in cat_list]
    return bbox_list, new_cat_list


result = None
for item in tqdm(images):
    mem = psutil.virtual_memory()
    kx = float(mem.available) / 1024 / 1024 / 1024
    if kx < 0.5:
        break
    img_name = item['file_name']
    if img_name not in errors or img_name in exist:
        continue
    bbox, label = get_bbox(item['id'])
    img = prefix + item['file_name']
    result = inference_detector(model, img)
    # print(result)
    model.show_result(img, result, out_file=save_dir + img_name.replace(".jpg","1.jpg"), bbox=bbox, label=label)
    del result, bbox, label, img_name, img
    gc.collect()
