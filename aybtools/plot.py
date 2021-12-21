# -*- coding:GBK -*-
"""
@Time ： 2021/11/12 15:26
@Auth ： aiyubin
@email : aiyubin1999@vip.qq.com
@File ：plot.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)

"""
"""plot feature map ,before use it map the feature to 1 channel and squeeze it."""
import numpy as np
from PIL import Image
import os


def plot(Path):
    if os.path.exists(Path.replace(".txt", ".jpg")):
        return
    f = open(Path, "r")
    line = f.readline()
    line = eval(line)
    f.close()
    line=np.array(line)
    temp = 255 / (line.max() - line.min())
    line = line - line.min()
    a = line * temp
    # print(a.mean())
    a = a.astype(np.uint8)
    b = Image.fromarray(a.squeeze(), mode='L')
    b.save(Path.replace(".txt", ".jpg"))


prefix = "/home/ayb/work_dirs/"
L = os.listdir(prefix)
for name in L:
    if ".txt" in name:
        Path = os.path.join(prefix, name)
        plot(Path)
        print("{} is ok!".format(os.path.join(prefix, name)))
