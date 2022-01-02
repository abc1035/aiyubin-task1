"""this file is aimed to generate a small datasets for test"""
import json

f = open("/home/ayb/UVM_Datasets/voc_test3.json", "r")
line = f.readline()
dic = eval(line)
f.close()
f2=open("/home/ayb/work_dirs/depth/wrong1.txt","r")
imgs=[]
while True:
    line=f2.readline()
    if line:
        imgs.append(line.rstrip())
    else:
        break
print(imgs)
f2.close()
images = dic['images']
new_images=[]
for image in images:
    if image['file_name'] not in imgs:
        continue
    else:
        new_images.append(image)
image_id = []
annotations = dic['annotations']
new_annotations = []
for image in new_images:
    # print(image)
    image_id.append(image['id'])
for annotation in annotations:
    if annotation['image_id'] in image_id:
        new_annotations.append(annotation)
dic["images"] = new_images
dic["annotations"] = new_annotations
f1 = open("/home/ayb/UVM_Datasets/voc_test_for_depth.json", "w")
dic_json = json.dumps(dic)
f1.write(str(dic_json))
f1.close()