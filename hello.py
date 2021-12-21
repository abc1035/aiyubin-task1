f1=open("/root/origin1/atss_r50_fpn_1x_coco1_from12.out","r")
L=[]
while True:
    line=f1.readline()
    if line:
        L.append(line)
    else :
        break
f1.close()
f2=open("/root/origin1/atss_r50_fpn_1x_coco1_from12_process.out","w")
for line in L:
    if "[" in line and  ">" in line and "]" in line:
        continue
    if "ETA" in line and "task/s" in line:
        continue
    else :
        f2.write(line)
f2.close()