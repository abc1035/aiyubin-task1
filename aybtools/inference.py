"""this file is aimed to inference a image and show its result"""

from mmdet.apis import init_detector, inference_detector
import mmcv

# Specify the path to model config and checkpoint file
config_file = '/home/ayb/origin1/configs/atss/atss_r34_two_fuse1.py'
checkpoint_file = '/home/ayb/work_dirs/fuse7_after_fpn/epoch_14.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = '/home/ayb/UVM_Datasets/voc8/VOCdevkit/VOC2007/JPEGImages/15_312.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# print(result)
# visualize the results in a new window
model.show_result(img, result)
# or save the visualization results to image files
model.show_result(img, result, out_file='home/ayb/result.jpg')

# test a video and show the results
