import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np
IMAGE_FOLDER = "/public_bme/data/jianght/datas/Pathology"

import os  
from PIL import Image  
import torch  
import torchvision.transforms as transforms  
from torch.utils.data import Dataset  
from glob import glob
from einops.layers.torch import Rearrange
  
        
class MultiDataSet(Dataset):
    def __init__(self, data, transforms=None, head_idx=None, age=False, img_batch=25,tasks=['fungus','label'],need_patch=False,patch_size=256):
        if isinstance(data,str) and  os.path.isfile(data):
            data = pd.read_csv(data)
        self.data = data
        self.head_idx = head_idx
        self.age = age
        self.img_batch = img_batch
        self.tasks = tasks
        self.patch_size = patch_size
        self.need_patch = need_patch


        if isinstance(self.tasks,str):
            self.tasks = [tasks]
        # print(self.data.columns.array)
        for i in self.tasks:
            assert i in self.data.columns.array, f'task names wrong get {i} ---- '

    def __getitem__(self, index):
        # Initialize transform and normalize


        # Read images
        folder_path = get_folder_path(self.data.iloc[index, 0])
        image_filenames = sorted(glob(f'{folder_path}/*.jpg'), key=lambda x: os.path.getsize(x), reverse=True)[:self.img_batch]
        # print('sss',folder_path)
        images = []
        for img_name in image_filenames:
            image_path = img_name
            image = Image.open(image_path)
            
            if self.need_patch:
                transform = transforms.Compose([
                #transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                Rearrange('c (h p1) (w p2) -> (h w) c p1 p2 ', p1=self.patch_size, p2=self.patch_size),
                ])
                image = transform(image)
                images.extend(image)
            else:
                transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                #Rearrange('c (h p1) (w p2) -> (h w) c p1 p2 ', p1=self.patch_size, p2=self.patch_size),
                ])
                image = transform(image)
                images.append(image)

            # Stack images
        images_tensor = torch.stack(images)
        # images_tensor = 1
        label_dict={}
        for i in range(len(self.tasks)):
            column_index = self.data.columns.get_loc(self.tasks[i])
            label_dict[f'label_{i}'] = self.data.iloc[index,column_index]
        if 'code' in self.data.columns:
            idx = self.data.columns.get_loc('code')
            label_dict['code'] = self.data.iloc[index,idx]
        else:
            label_dict['code'] = self.data.iloc[index, 0].split('/')[-1]
        if 'highlabel' in self.data.columns:
            multilabel_index = self.data.columns.get_loc('highlabel')
        else:
            multilabel_index = self.data.columns.get_loc('multilabel')
        label_dict['multilabel'] = self.data.iloc[index, multilabel_index]
        labels = label_dict

        if self.head_idx is not None:
            # print(labels[f'label_{self.head_idx}'],type(labels[f'label_{self.head_idx}']))
            return images_tensor, labels[f'label_{self.head_idx}']
        else:
            return images_tensor, labels

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
    

def get_folder_path(name):

    folder_path = ''
    if os.path.isdir(os.path.join(IMAGE_FOLDER, name, 'torch')):
        folder_path = os.path.join(IMAGE_FOLDER, name, 'torch')
    elif os.path.isdir(os.path.join(IMAGE_FOLDER,'yangxing','Torch',name,'torch')):
        folder_path = os.path.join(IMAGE_FOLDER,'yangxing','Torch',name,'torch')
    elif os.path.isdir(os.path.join(IMAGE_FOLDER,'yinxing','Torch',name,'torch')):
        folder_path = os.path.join(IMAGE_FOLDER,'yinxing','Torch',name,'torch')
    elif os.path.isdir(os.path.join(IMAGE_FOLDER, name)):
        folder_path = os.path.join(IMAGE_FOLDER, name)
    
    assert os.path.isdir(folder_path),f'get wrong name {name}'
    
    return folder_path
if __name__=='__main__':
    # data1=Gongjing('D:\\Datas\\bingli')
    # data2=Fungus('E:\\fungus')
    data3 = MultiDataSet(data='/public_bme/data/jianght/datas/Pathology/class2/test_supplement3.csv')
    # data=data1+data2
    data = data3
    loader=torch.utils.data.DataLoader(data,shuffle=True,batch_size=4)

    for i,l in loader:
        print(i.size(),l)


