import os

base_path = '/public_bme/data/jianght/datas/Pathology/yinxing/Torch'

for path1 in os.listdir(base_path):
    basepath1 = os.path.join(base_path,path1)
    for path2 in os.listdir(basepath1):
        filepath = os.path.join(basepath1,path2,'torch')
        files = os.listdir(filepath)
        if len(files) != 100:
            print(filepath,len(files))
