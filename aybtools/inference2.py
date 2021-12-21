# -*- coding:GBK -*-
"""
@Time ： 2021/11/12 15:26
@Auth ： aiyubin
@email : aiyubin1999@vip.qq.com
@File ：{name}.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)

"""
"""inference all images and find different with gt"""
import os
from mmdet.apis import init_detector, inference_detector
import mmcv
from tqdm import tqdm
import numpy as np
import torch

"""import torch.multiprocessing as mp
from torch.multiprocessing import Process, set_start_method"""

prefix = "/home/ayb/UVM_Datasets/voc8/VOCdevkit/VOC2007/JPEGImages/"
L = os.listdir(prefix)
config_file = '/home/ayb/origin1/configs/atss/atss_r34_nonlocal.py'
checkpoint_file = "/home/ayb/work_dirs/nonlocal/epoch_22.pth"
img_dir = '/home/ayb/work_dirs/nonlocal/'
model = init_detector(config_file, checkpoint_file, device='cuda:0')
object_total = 0
correct_total = 0


def fuck(img):
    img_name = img.split('/')[-1]
    result = inference_detector(model, img)
    # visualize the results in a new window
    # model.show_result(img, result)
    # or save the visualization results to image files
    # model.show_result(img, result, out_file=img_dir + img_name)
    bboxes, labels = model.show_result(img, result)
    return bboxes, labels


def load_annotations(ann_path="/home/ayb/UVM_Datasets/voc_test3.json"):
    f = open(ann_path)
    dic = f.readline()
    dic = eval(dic)
    f.close()
    return dic


def find_bbox(annotations, image_id):
    L = []
    for annotation in annotations:
        if annotation['image_id'] == image_id:
            L.append(annotation)
    return L


def predict(file_name):
    # mutex.acquire()
    bbox, labels = fuck(prefix + file_name)
    # mutex.release()
    return bbox, labels


def check(bbox_list, pre_bbox, pre_label):
    """pre_bbox (list) : one item is [x1,y1,x2,y2,score]"""
    bboxes = []
    labels = []
    for bbox in bbox_list:
        bboxes.append(bbox['bbox'])
        labels.append(bbox['category_id'])
    new_labels = labels
    labels = np.array(labels)
    labels -= 1
    new_labels.sort()
    new_labels = np.array(new_labels)
    new_labels -= 1
    bboxes = np.array(bboxes)
    num_bbox = bboxes.shape[0]
    global object_total
    object_total += num_bbox
    """print(new_labels)
    print(pre_label)
    print(new_labels == pre_label)"""
    # print(np.all(new_labels == pre_label))
    """if not np.all(new_labels == pre_label):  # label is not corresponding
        return False"""
    pre_list = [i for i in range(num_bbox)]
    gt_list = [i for i in range(num_bbox)]
    while True:
        if len(pre_list) <= 0:
            return True
        judge = False
        score = pre_bbox[pre_list[0]][-1]
        if score < 0.3:
            pre_list.remove(pre_list[0])
            continue
        for index in gt_list:
            temp = pre_bbox[pre_list[0]][:-1]
            temp1 = torch.tensor(np.expand_dims(temp, 0))
            bbox_temp = bboxes[index]
            bbox_temp = bbox_temp.tolist()
            bbox_temp[2] = bbox_temp[0] + bbox_temp[2]
            bbox_temp[3] = bbox_temp[1] + bbox_temp[3]
            bbox_temp = np.array(bbox_temp)
            temp2 = torch.tensor(np.expand_dims(bbox_temp, 0))
            # print()
            overlap = bbox_overlaps(temp1, temp2)
            # print(overlap.sum())
            if labels[index] == pre_label[pre_list[0]] and overlap.sum() > 0.75:
                gt_list.remove(index)
                judge = True
                global correct_total
                correct_total += 1
                break
        pre_list.remove(pre_list[0])
        # print(judge)
        if not judge:
            return False


def find_mistakes(dic, images):
    f = open("/home/ayb/work_dirs/nonlocalerror/wrong1.txt", "w")
    # cnt = 0
    annotations = dic['annotations']
    for image in tqdm(images):
        file_name = image['file_name']
        image_id = image['id']
        bbox_list = find_bbox(annotations, image_id)
        """print(bbox_list)"""
        pre_bbox, pre_label = predict(file_name)
        judge = False
        try:
            judge = check(bbox_list, pre_bbox, pre_label)
        except BaseException:
            judge = False
            pass
        if not judge:
            # falseimage.put(file_name)
            f.write(file_name + "\n")
            # cnt += 1
        """if images.qsize() % 100 == 0:
            print(images.qsize)"""
    # print("total:{}".format(cnt))
    f.close()


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.

    FP16 Contributed by https://github.com/open-mmlab/mmdetection/pull/4889
    Note:
        Assume bboxes1 is M x 4, bboxes2 is N x 4, when mode is 'iou',
        there are some new generated variable when calculating IOU
        using bbox_overlaps function:

        1) is_aligned is False
            area1: M x 1
            area2: N x 1
            lt: M x N x 2
            rb: M x N x 2
            wh: M x N x 2
            overlap: M x N x 1
            union: M x N x 1
            ious: M x N x 1

            Total memory:
                S = (9 x N x M + N + M) * 4 Byte,

            When using FP16, we can reduce:
                R = (9 x N x M + N + M) * 4 / 2 Byte
                R large than (N + M) * 4 * 2 is always true when N and M >= 1.
                Obviously, N + M <= N * M < 3 * N * M, when N >=2 and M >=2,
                           N + 1 < 3 * N, when N or M is 1.

            Given M = 40 (ground truth), N = 400000 (three anchor boxes
            in per grid, FPN, R-CNNs),
                R = 275 MB (one times)

            A special case (dense detection), M = 512 (ground truth),
                R = 3516 MB = 3.43 GB

            When the batch size is B, reduce:
                B x R

            Therefore, CUDA memory runs out frequently.

            Experiments on GeForce RTX 2080Ti (11019 MiB):

            |   dtype   |   M   |   N   |   Use    |   Real   |   Ideal   |
            |:----:|:----:|:----:|:----:|:----:|:----:|
            |   FP32   |   512 | 400000 | 8020 MiB |   --   |   --   |
            |   FP16   |   512 | 400000 |   4504 MiB | 3516 MiB | 3516 MiB |
            |   FP32   |   40 | 400000 |   1540 MiB |   --   |   --   |
            |   FP16   |   40 | 400000 |   1264 MiB |   276MiB   | 275 MiB |

        2) is_aligned is True
            area1: N x 1
            area2: N x 1
            lt: N x 2
            rb: N x 2
            wh: N x 2
            overlap: N x 1
            union: N x 1
            ious: N x 1

            Total memory:
                S = 11 x N * 4 Byte

            When using FP16, we can reduce:
                R = 11 x N * 4 / 2 Byte

        So do the 'giou' (large than 'iou').

        Time-wise, FP16 is generally faster than FP32.

        When gpu_assign_thr is not -1, it takes more time on cpu
        but not reduce memory.
        There, we can reduce half the memory and keep the speed.

    If ``is_aligned`` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned`` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )

    Example:
        >>> empty = torch.empty(0, 4)
        >>> nonempty = torch.FloatTensor([[0, 0, 10, 9]])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows,))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
            bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
            bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2],
                       bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:],
                       bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                    bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                    bboxes2[..., None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


def fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        return x.float().clamp(min, max).half()

    return x.clamp(min, max)


def inference_all(images):
    for image in tqdm(images):
        file_name = image['file_name']
        pre_bbox, pre_label = predict(file_name)
        pre_dic[file_name] = (pre_bbox, pre_label)
    return pre_dic


if __name__ == "__main__":
    # set_start_method('spawn')
    dic = load_annotations()
    images = dic['images']
    find_mistakes(dic, images)
    print("total:{},correct:{},percent:{}".format(object_total, correct_total, correct_total / object_total))
