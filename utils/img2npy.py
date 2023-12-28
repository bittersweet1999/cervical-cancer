from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os
from PIL import Image
from glob import glob
import torch
from torchvision import transforms
from time import sleep
from tqdm import tqdm

IMAGE_FOLDER = '/public_bme/data/jianght/datas/Pathology/20231108_bumanyi_150'
SAVE_FOLDER = '/public_bme/data/jianght/datas/Pathology_256_pt/20231108_bumanyi_150'

transform = transforms.Compose([
transforms.Resize((256, 256)),
transforms.ToTensor(),
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#Rearrange('c (h p1) (w p2) -> (h w) c p1 p2 ', p1=self.patch_size, p2=self.patch_size),
])

def img2tensor(root):
    
    files1 = glob(f'{root}/*.jpg')
    files1 = sorted(files1, key=lambda x: os.path.getsize(x), reverse=True)[:200]
    save_dir = root.replace(IMAGE_FOLDER,SAVE_FOLDER)
    save_name = os.path.join(save_dir,'images.pt')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    if os.path.isfile(save_name):
        data = torch.load(save_name)
        if data.shape[0] == len(files1):
            # print(f'{save_dir} exist -------------')
            return
        
    img_list = []
    for file in files1:
        img = Image.open(file)
        img_tensor = transform(img)
        img_list.append(img_tensor)
    img_tensors = torch.stack(img_list)

    assert img_tensors.shape[0] == min(len(files1),200)
    
    torch.save(img_tensors,save_name)
    print(save_name,img_tensors.shape)

def Img2TensorThread():
    executor = ThreadPoolExecutor(max_workers=64)
    for root,dir,files in tqdm(os.walk(IMAGE_FOLDER)):
        if len(files) >50:
            executor.submit(img2tensor, root )
    executor.shutdown(wait=True)
    print('finish----------------------------------------------')
    
Img2TensorThread()

# if __name__ == '__main__':
#     for root,dir,files in os.walk(IMAGE_FOLDER):
#         if len(files) >50:
#             img2tensor(root)
            # executor.submit(img2tensor, root )
    



