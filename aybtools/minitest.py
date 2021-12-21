"""this file is aimed to generate a small datasets for test"""
import json

f = open("/home/ayb/UVM_Datasets/voc_test3.json", "r")
line = f.readline()
f.close()
dic = eval(line)
images = dic['images']
new_images=[]
for image in images:
    if "ten" in image['file_name']:
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
f1 = open("/home/ayb/UVM_Datasets/voc_test_not_ten.json", "w")
dic_json = json.dumps(dic)
f1.write(str(dic_json))
f1.close()