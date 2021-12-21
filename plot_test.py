# -*- coding:GBK -*-
"""
@Time ： 2021/11/12 15:26
@Auth ： aiyubin
@email : aiyubin1999@vip.qq.com
@File ：{name}.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)

"""
import numpy as np
from PIL import Image

a = np.ones([2550, 1920])
"""for i in range(255):
    a[10 * i:10*i + 10, :] = i"""

a = a * 4
a = a.astype(np.uint8)
b = Image.fromarray(a.squeeze(), mode='L')
b.show()
