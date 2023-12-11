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
  
class MyDataSet(Dataset):  
  
    def __init__(self, data, soft_labels_filename=None, transforms=None,head_idx=None,age=False, img_batch=25):
        self.data = data  
        self.transforms = transforms
        self.head_idx=head_idx
        self.age=age
        self.img_batch=img_batch
  
    def __getitem__(self, index):  
        # Initialize transform and normalize  
        transform = transforms.Compose([  
            transforms.Resize((224, 224)),  
            transforms.ToTensor(),  
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
        ])  
  
        # Read images  
        folder_path = os.path.join(IMAGE_FOLDER, self.data.iloc[index, 0],'torch')  
        image_filenames = sorted(glob(f'{folder_path}/*.jpg'), key=lambda x: os.path.getsize(x), reverse=True)[:self.img_batch]
        images = []
        for img_name in image_filenames:
            image_path = img_name
            image = Image.open(image_path)  
            image = transform(image)  
            images.append(image)  
  
        # Stack images  
        images_tensor = torch.stack(images)  
  
        label_1 = self.data.iloc[index, 1]
        if self.age:
            label_0 = self.data.iloc[index, 2]
        else:
            label_0 = 0

        labels={
            'label_0': label_0,
            'label_1': label_1,
            'belong' : 'Gongjing'
        }

        #return images_tensor,label_1

        if self.head_idx is not None:
            # print(labels[f'label_{self.head_idx}'],type(labels[f'label_{self.head_idx}']))
            return images_tensor,labels[f'label_{self.head_idx}']
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


class Fungus(Dataset):
    def __init__(self, data, img_batch=100,head_idx=None):
        self.data = data
        data_paths = os.listdir(data)
        self.img_lists=[]
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.transform=transform
        self.img_batch=img_batch
        for path in data_paths:
            img_path = os.path.join(data,path,'torch')
            imgs = glob(f'{img_path}/*.jpg')
            imgs = sorted(imgs,key=lambda x: os.path.getsize(x) ,reverse=True)
            # img_datas = [ imgs[ i : i+self.img_batch ] for i in range(0,250,25)]
            img_datas = imgs[:self.img_batch]
            self.img_lists.append(img_datas)

        self.head_idx=head_idx

        # print(self.img_lists[:2])
        print(len(self.img_lists),len(self.img_lists[0]),len(data_paths))
    def __len__(self):
        return len(self.img_lists)

    def __getitem__(self, item):
        image_filenames = self.img_lists[item]
        images = []
        for image_path in image_filenames:
            image = Image.open(image_path)
            image = self.transform(image)
            images.append(image)

        images_tensor = torch.stack(images)
        # labels= torch.ones(len(image_filenames))
        labels={
            'label_0' : 1,
            'label_1' : 0,
            'belong'  : 'Fungus'
        }
        # labels=torch.Tensor([
        #     [1],[0]
        # ])
        
        #return images_tensor,labels['label_1']
        
        if self.head_idx is not None:
            return images_tensor,labels[f'label_{self.head_idx}']
        else:
            return images_tensor,labels
    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

class Gongjing(Dataset):
    def __init__(self,data,img_batch=100,head_idx=None):
        super().__init__()
        label_paths=os.listdir(data)
        self.img_lists=[]
        for label_path in label_paths:
            if 'yin' in label_path or 'yang' in label_path:
                self.img_lists.extend(glob(f'{os.path.join(data,label_path)}/*'))
        self.img_batch=img_batch
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.transform=transform
        self.head_idx=head_idx
        print(len(self.img_lists))

    def __len__(self):
        return len(self.img_lists)

    def __getitem__(self, item):
        img_list=self.img_lists[item]+'/torch'
        label=0 if 'yin' in img_list else 1
        imgs=[]
        # t1=time.time()
        for img_path in os.listdir(img_list)[:self.img_batch]:
            img=Image.open(os.path.join(img_list,img_path))
            # img=img.resize((256,256))
            if self.transform:
                img=self.transform(img)
            imgs.append(img)
        imgs=torch.stack(imgs,0)
        # print(imgs.size())

        labels={
            'label_0' : 0,
            'label_1' : label,
            'belong'  : 'Gongjing'
        }
        # labels=torch.Tensor([
        #     [0],[label]
        # ])
        # t2=time.time()
        # print(t2-t1)
        if self.head_idx is not None:
            return imgs,labels[f'label_{self.head_idx}']
        else:
            return imgs,labels
        # return imgs,labels
    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
        
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
        print(self.data.columns.array)
        for i in self.tasks:
            assert i in self.data.columns.array, f'task names wrong get {i} ---- '

    def __getitem__(self, index):
        # Initialize transform and normalize


        # Read images
        # if os.path.isdir(os.path.join(IMAGE_FOLDER, self.data.iloc[index, 0], 'torch')):
        #     folder_path = os.path.join(IMAGE_FOLDER, self.data.iloc[index, 0], 'torch')
        # else:
        #     folder_path = os.path.join(IMAGE_FOLDER, self.data.iloc[index, 0])
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


