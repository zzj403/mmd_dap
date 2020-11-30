from pycocotools.coco import COCO

import matplotlib.pyplot as plt
import cv2

import os
import numpy as np
import random
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
from skimage import io,transform
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms
import numpy as np
import PIL.Image as Image








class CocoValPerson(Dataset): #继承Dataset
    def __init__(self, cocoRoot="/disk2/mycode/common_data/coco", dataType="val2017", num_use=None): #__init__是初始化该类的一些基础参数

        self.cocoRoot = cocoRoot
        self.dataType = dataType

        annFile = os.path.join(self.cocoRoot, f'annotations/instances_{self.dataType}.json')
        print(f'Annotation file: {annFile}')


        self.coco=COCO(annFile)

        # 利用getCatIds函数获取某个类别对应的ID，
        # 这个函数可以实现更复杂的功能，请参考官方文档
        person_id = self.coco.getCatIds('person')[0]
        print(f'"person" 对应的序号: {person_id}')

        # 利用loadCats获取序号对应的文字类别
        # 这个函数可以实现更复杂的功能，请参考官方文档
        cats = self.coco.loadCats(1)
        print(f'"1" 对应的类别名称: {cats}')


        self.imgIds = self.coco.getImgIds(catIds=[1])
        print(f'包含person的图片共有：{len(self.imgIds)}张')

        # crowds filter
        new_imgIds = []
        for i in range(len(self.imgIds)):
            imgId = self.imgIds[i]
            annIds = self.coco.getAnnIds(imgIds=imgId, catIds=[1], iscrowd=True)
            if len(annIds) == 0:
                new_imgIds.append(imgId)
        self.imgIds = new_imgIds
        print(f'筛选掉crowds mask 的图片后，剩余：{len(self.imgIds)}张')

        if num_use != None:
            self.imgIds = self.imgIds[:num_use]
            print(f'Only use {num_use} images')



    
    def __len__(self):
        return len(self.imgIds)
    
    def __getitem__(self, index):

        imgId = self.imgIds[index]
        imgInfo = self.coco.loadImgs(imgId)[0]
        imPath = os.path.join(self.cocoRoot, self.dataType, imgInfo['file_name'])

        img = Image.open(imPath).convert('RGB')
        img = transforms.Resize((500, 500))(img)
        img = transforms.ToTensor()(img)

        annIds = self.coco.getAnnIds(imgIds=imgId, catIds=[1])

        anns = self.coco.loadAnns(annIds)



        masks_tensor = torch.Tensor(14,500,500).fill_(-1)
        box_tesnor = torch.Tensor(14,4).fill_(-1)
        h_w_r_tensor = torch.Tensor(14).fill_(-1)

        one_layer = torch.ones(1,500,500)
        zero_layer = torch.zeros(1,500,500)


        if len(annIds) >= 14:
            print(imgInfo['file_name'])
            # print(len(annIds))

        for i in range(len(annIds)):
            if anns[i]['iscrowd'] == 1:
                print(imgInfo['file_name'])
                print(len(annIds))
                continue

            mask = self.coco.annToMask(anns[i])
            mask = torch.from_numpy(mask).float()
            mask = transforms.ToPILImage()(mask)
            mask = transforms.Resize((500, 500))(mask)
            mask = transforms.ToTensor()(mask)
            mask = torch.where(mask>0.5, one_layer, zero_layer)
            masks_tensor[i] = mask

            box = anns[i]['bbox']

            h_w_r = box[3]/box[2]

            box_trans = box.copy()

            box_trans[0] = box[0]/imgInfo['width'] * 500
            box_trans[1] = box[1]/imgInfo['height'] * 500
            box_trans[2] = box[2]/imgInfo['width'] * 500
            box_trans[3] = box[3]/imgInfo['height'] * 500

            
            box_tesnor[i] = torch.Tensor(box_trans)
            h_w_r_tensor[i] = h_w_r



        # masks_area_sort_index = torch.sort(masks_area_tensor, descending=True)[1]
        # masks_tensor_sort = masks_tensor[masks_area_sort_index]
        # vali = torch.sum(torch.sum(masks_tensor_sort, dim=-1), dim=-1)

        # masks_tensor_sort_top = masks_tensor_sort[:14]
        # masks_tensor_sort_top_len = masks_tensor_sort_top.shape[0]

        # masks_tensor_return = torch.Tensor(14,1,500,500).fill_(-1)

        # masks_tensor_return[:masks_tensor_sort_top_len] = masks_tensor_sort[:masks_tensor_sort_top_len]


        # if len(annIds) >= 14:
        #     mask = masks_tensor_return[0]
        #     mask = transforms.ToPILImage()(mask)
        #     mask.show()

        return img, masks_tensor, box_tesnor, h_w_r_tensor

if __name__=='__main__':
    data = CocoValPerson(dataType="val2017", num_use=10)
    dataloader = DataLoader(data, batch_size=1,shuffle=False) #使用DataLoader加载数据

    max_len = 0
    for epoch in range(10):
        for i_batch,batch_data in enumerate(dataloader):
            if i_batch % 50 ==0:
                
                img, masks, bboxes, h_w_r = batch_data
                # masks_pil = transforms.ToPILImage()(masks[0,0])
                # masks_pil.show()
                bbox = bboxes[0,0]
                cccc = masks[0,0].clone()
                cccc[int(bbox[1]):int(bbox[1]+bbox[3]),int(bbox[0]):int(bbox[0]+bbox[2])] = 1
                cccc_p = cccc+masks[0,0]
                cccc_p = cccc_p/torch.max(cccc_p)
                cccc_p_pil = transforms.ToPILImage()(cccc_p)
                cccc_p_pil.show()




                print(i_batch)
        
       

