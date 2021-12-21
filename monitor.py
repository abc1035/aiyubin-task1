from pynvml import *
import time
import os
import smtplib
from email.mime.text import MIMEText
from email.header import Header

gpu_map = {0: 0, 1: 1, 2: 6, 3: 7, 4: 2, 5: 3, 6: 4, 7: 5}
gpu_map_reverse = {0: 0, 1: 1, 6: 2, 7: 3, 2: 4, 3: 5, 4: 6, 5: 7}
need_space = 8000
deploy_path = "/root/mmdetection/configs/yolox/yolox_s_8x8_300e_coco.py"
work_dir = "/root/mmdetection/work_dirs/yolox_s_8x8_300e_coco"
command = "nohup python tools/train.py {}  --gpu-ids={} >yunxing{}.out  &"
samples_per_gpu = 6
workers_per_gpu = 1


def gpu_info():
    infor = {}
    status = True
    status_list = []
    nvmlInit()
    count = nvmlDeviceGetCount()
    for i in range(count):
        handle = nvmlDeviceGetHandleByIndex(i)
        name = nvmlDeviceGetName(handle)
        info = nvmlDeviceGetMemoryInfo(handle)
        status_list.append([i, info.used / 1024 / 1024, info.free / 1024 / 1024])
        # status_list.append(info.used / info.total)
    nvmlShutdown()
    '''for i in status_list:
        if i >= risk:
            status=False
    '''
    return status_list


def gpu_info_index_mem(i):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(i)
    info = nvmlDeviceGetMemoryInfo(handle)
    nvmlShutdown()
    return info.info.free / 1024 / 1024


def update_file():
    f1 = open(deploy_path, "r")
    lines = []
    while True:
        line = f.readline()
        if line:
            lines.append(line)
        else:
            break
    f1.close()
    f2 = open(deploy_path, "w")
    for line in lines:
        if line[:20] == "    samples_per_gpu=":
            f2.write(line[:20] + str(samples_per_gpu) + ",\n")
        elif line[:20] == "    workers_per_gpu=":
            f2.write(line[:20] + str(workers_per_gpu) + ",\n")
        else:
            f2.write(line)
    f2.close()
    return


def get_param(gpu_id):
    config_path = deploy_path
    points = os.listdir(work_dir)
    checkpoint = False
    resume = ""
    maxid = 0
    resume_name = ""
    for point in points:
        if "epoch" in point and ".pth" in point:  ## checkpoint
            checkpoint = True
            id = int(point.split('_')[1].split('.')[0])
            if id > maxid:
                resume_name = point
                maxid = id
    if checkpoint:
        resume = "--resume-from {}".format(os.path.join(work_dir, resume_name))
    return [config_path, resume, gpu_id, maxid + 1]
def send_message(gpu_id):
    f=open("log.txt","a")
    f.write("gpu-ids={},samples_per_gpu={}".format(gpu_map_reverse[gpu_id],samples_per_gpu))
    f.close()

def execute(gpu_id, mem_free):
    update_file()
    param = get_param(gpu_id)
    while True:
        os.system(command.format(param[0], param[1], param[2], param[3]))
        time.sleep(30)
        gpu_new_free = gpu_info_index_mem(gpu_map_reverse[gpu_id])
        if mem_free - gpu_new_free > 3000:  # success
            send_message(gpu_id)
            samples_per_gpu=6
            return
        else:
            samples_per_gpu -= 1
            execute(gpu_id, mem_free)


def monitor():
    while True:
        status_list = gpu_info()
        for i, gpu in enumerate(status_list):
            if gpu[3] > need_space:
                execute(gpu_map[i], gpu[3])
            else:
                continue
        time.sleep(3)
