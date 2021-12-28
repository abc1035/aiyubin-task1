# -*- coding:GBK -*-
"""
@Time ： 2021/11/12 15:26
@Auth ： aiyubin
@email : aiyubin1999@vip.qq.com
@File ：{name}.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)

"""

"""this function was used to calculte the bbox area/total area,cause i wan't to use focal loss in depth."""
f=open("/home/ayb/UVM_Datasets/voc_train3.json")
line=f.readline()
line=eval(line)
images=line['images']
anns=line['annotations']
imagetotal=0
bboxtotal=0
for image in images:
    try:
        imagetotal+=image['height']*image['width']
    except BaseException:
        continue
for ann in anns:
    try:
        bboxtotal+=ann['bbox'][2]*ann['bbox'][3]
    except BaseException:
        continue
print("bbox:{} image:{} bbox/image:{}".format(bboxtotal,imagetotal,bboxtotal/imagetotal))
