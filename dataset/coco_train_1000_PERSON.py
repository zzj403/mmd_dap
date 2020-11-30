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
from skimage import measure
from tqdm import tqdm

class CocoTrainPerson(Dataset): #继承Dataset
    def __init__(self, cocoRoot="../common_data/coco", dataType="train2017", num_use=None): #__init__是初始化该类的一些基础参数

        self.CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
        self.cocoRoot = cocoRoot
        self.dataType = dataType

        annFile = os.path.join(self.cocoRoot, f'annotations/instances_{self.dataType}.json')
        print(f'Annotation file: {annFile}')


        self.coco=COCO(annFile)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

        # 利用getCatIds函数获取某个类别对应的ID，
        # 这个函数可以实现更复杂的功能，请参考官方文档
        # person_id = self.coco.getCatIds('person')[0]
        # print(f'"person" 对应的序号: {person_id}')

        # 利用loadCats获取序号对应的文字类别
        # 这个函数可以实现更复杂的功能，请参考官方文档
        # cats = self.coco.loadCats(1)
        # print(f'"1" 对应的类别名称: {cats}')

        # self.imgIds_all = self.coco.getImgIds()
        self.imgIds_all = self.coco.getImgIds(catIds=[1])
        print(f'包含person的图片共有：{len(self.imgIds_all)}张')

        
        # annIds = self.coco.getAnnIds()
        # anns = self.coco.loadAnns(annIds)
        # want big object
        new_annoIds = []
        new_imgIds = []

        save_path = os.path.join('../common_data/coco/zzj_choose', dataType+'_select.npy')
        if os.path.exists(save_path):
            save_annoIds_np = np.load(save_path)
            new_annoIds = save_annoIds_np.tolist()
            print('load selected annos from saved npy')
        else:
            for i in tqdm(range(len(self.imgIds_all))):
                imgId = self.imgIds_all[i]
                annIds = self.coco.getAnnIds(imgIds=imgId, catIds=[1])
                anns = self.coco.loadAnns(annIds)

                imgInfo = self.coco.loadImgs(imgId)[0]
                img_area = imgInfo['width'] * imgInfo['height']
                img_w = imgInfo['width']
                img_h = imgInfo['height']
                aspect_ratio = max(img_w, img_h) / min(img_w, img_h)

                if aspect_ratio > 1.4:
                    continue

                for j in range(len(anns)):
                    ann = anns[j]
                    if ann['iscrowd']:
                        continue
                    mask_area = ann['area']
                    mask_area_ratio = mask_area / img_area
                    if mask_area_ratio > 0.4 and mask_area_ratio < 0.6:
                        mask = self.coco.annToMask(ann)
                        labels = measure.label(mask, background=0, connectivity=2)
                        if np.max(labels)>1:
                            continue
                        new_annoIds.append(annIds[j])
                        new_imgIds.append(imgId)
            print(len(new_annoIds))



            # for img_id in tqdm(new_imgIds):
            #     imgInfo = self.coco.loadImgs(img_id)[0]
            #     name = imgInfo['file_name']
            #     source = os.path.join('/disk2/mycode/common_data/coco/train2017', name)
            #     path = os.path.join('/disk2/mycode/common_data/coco/zzj_choose/train_img_choose', name)
            #     if not os.path.exists(path):

            #         os.system('cp '+ source +' '+path)

            save_annoIds_np = np.array(new_annoIds)
            np.save(save_path, save_annoIds_np)

            # new_imgIds_np = np.array(new_imgIds)
            # new_imgIds_np = np.unique(new_imgIds_np)
            # new_imgIds = new_imgIds_np.tolist()
            # all_anns = self.coco.loadAnns(new_annoIds)

            # imgIds = new_imgIds
            # self.annoIds = new_annoIds
            # print(len(new_imgIds))


        self.annoIds = new_annoIds




        print('After select, annos remains',len(new_annoIds))
        if num_use != None:
            self.annoIds = self.annoIds[:num_use]
            print(f'Only use {num_use} objects')



    
    def __len__(self):
        return len(self.annoIds)
    
    def __getitem__(self, index):

        annoId = self.annoIds[index]
        ann = self.coco.loadAnns(annoId)[0]
        imgId = ann['image_id']
        imgInfo = self.coco.loadImgs(imgId)[0]
        imPath = os.path.join(self.cocoRoot, self.dataType, imgInfo['file_name'])


        img = Image.open(imPath).convert('RGB')
        img = transforms.Resize((500, 500))(img)
        img = transforms.ToTensor()(img)



        one_layer = torch.ones(500,500)
        zero_layer = torch.zeros(500,500)


        mask = self.coco.annToMask(ann)
        mask = torch.from_numpy(mask).float()
        mask = transforms.ToPILImage()(mask)
        mask = transforms.Resize((500, 500))(mask)
        mask = transforms.ToTensor()(mask)
        mask = torch.where(mask>0.5, one_layer, zero_layer)
        mask = mask.squeeze()


        box = ann['bbox']
        box_trans = box.copy()

        box_trans[0] = box[0]/imgInfo['width'] * 500
        box_trans[1] = box[1]/imgInfo['height'] * 500
        box_trans[2] = box[2]/imgInfo['width'] * 500
        box_trans[3] = box[3]/imgInfo['height'] * 500

        
        box_tesnor = torch.Tensor(box_trans)
        class_ = self.cat2label[ann['category_id']]

        new_name = imgInfo['file_name'].split('.')[0]+'_'+str(annoId) + '.png'


        return img, mask, box_tesnor, class_, new_name

if __name__=='__main__':
    data = CocoTrainPerson( num_use=500)
    dataloader = DataLoader(data, batch_size=1,shuffle=False) #使用DataLoader加载数据

    max_len = 0
    for epoch in range(10):
        for i_batch, batch_data in enumerate(dataloader):
            if i_batch % 1 == 0:
                
                img, mask, bbox, class_ = batch_data[0][0], batch_data[1][0], batch_data[2][0], batch_data[3][0]
                # masks_pil = transforms.ToPILImage()(masks[0,0])
                # masks_pil.show()

                cccc = mask.clone()
                cccc[int(bbox[1]):int(bbox[1]+bbox[3]),int(bbox[0]):int(bbox[0]+bbox[2])] = 1
                cccc_p = cccc+mask
                cccc_p = cccc_p/torch.max(cccc_p)

                cccc_p  = (cccc_p + img)/2

                cccc_p_pil = transforms.ToPILImage()(img)
                cccc_p_pil.save('2k2.png')




                print(i_batch)
        
       

