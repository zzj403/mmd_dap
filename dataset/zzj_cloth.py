from torch.utils.data import DataLoader,Dataset
from skimage import io,transform
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms
import numpy as np
import PIL.Image as Image

class ClothData(Dataset): #继承Dataset
    def __init__(self, img_dir, mask_dir): #__init__是初始化该类的一些基础参数
        self.img_dir = img_dir   
        self.mask_dir = mask_dir 
        self.images = os.listdir(self.img_dir)
        self.masks = os.listdir(self.mask_dir)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_name = self.images[index] 
        # image_name = '0.png'
        img_path = os.path.join(self.img_dir, image_name)
        img = Image.open(img_path).convert('RGB')
        img = transforms.ToTensor()(img)

        mask_name = image_name
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = Image.open(mask_path)
        mask = 1 - transforms.ToTensor()(mask)[3].unsqueeze(0)

        return img, mask

if __name__=='__main__':
    data = ClothData(img_dir='/disk2/mycode/common_data/zzj_cloth/img500/',
                        mask_dir='/disk2/mycode/common_data/zzj_cloth/mask500/')
    dataloader = DataLoader(data, batch_size=4,shuffle=True) #使用DataLoader加载数据
    for i_batch,batch_data in enumerate(dataloader):
        img, mask = batch_data
        print(i_batch)#打印batch编号
        print(batch_data['image'].size())#打印该batch里面图片的大小
        print(batch_data['label'])#打印该batch里面图片的标签
