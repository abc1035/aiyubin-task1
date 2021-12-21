# -*- coding:GBK -*-
"""
@Time ： 2021/11/12 15:26
@Auth ： aiyubin
@email : aiyubin1999@vip.qq.com
@File ：{name}.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)

"""
"""find unique between two text"""
import os

img_dir = '/home/ayb/work_dirs/'
f1 = open("/home/ayb/work_dirs/fuseinfer/wrong.txt", "r")
f2 = open("/home/ayb/work_dirs/rgbinfer/wrong.txt", "r")
L1 = []
L2 = []
while True:
    line = f1.readline()
    if line:
        L1.append(line.rstrip())
    else:
        break
f1.close()
while True:
    line = f2.readline()
    if line:
        L2.append(line.rstrip())
    else:
        break
f2.close()
L3 = []
for pic in L2:
    if pic not in L1:
        L3.append(pic)
for pic in L3:
    os.system("cp -r {}/rgbinfer/{} {}/rgberror/{}".format(img_dir, pic, img_dir, pic))

print(L3)
