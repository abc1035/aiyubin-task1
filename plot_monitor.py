# -*- coding:GBK -*-
"""
@Time ： 2021/11/12 15:26
@Auth ： aiyubin
@email : aiyubin1999@vip.qq.com
@File ：{name}.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)

"""
import psutil
import os
import time

save_dir = '/home/ayb/work_dirs/nonlocalerror/'
error_path = "/home/ayb/work_dirs/nonlocalerror/wrong1.txt"
f = open(error_path, "r")
errors = []
while True:
    line = f.readline()
    if line:
        errors.append(line.rstrip())
    else:
        break
f.close()
exist = os.listdir(save_dir)


def judge():
    L = os.listdir(save_dir)
    L.remove("wrong1.txt")
    for error in errors:
        if error not in L:
            return False
    return True


if __name__ == "__main__":
    cnt = 0
    while True:
        mem = psutil.virtual_memory()
        kx = float(mem.available) / 1024 / 1024 / 1024
        print(kx)
        if judge():
            break
        if kx > 8:
            os.system("python aybtools/plotgt.py")
            cnt += 1
            print("run {}times".format(cnt))
        # time.sleep(5)
