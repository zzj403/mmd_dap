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
import torch.nn.functional as F
from skimage.morphology import convex_hull_image

class SuperPixelGet(Dataset): #继承Dataset
    def __init__(self, segments_label, segments_tensor, g_theta_m, data_num):
        self.segments_label = segments_label.cuda()
        self.segments_tensor = segments_tensor.cuda()
        self.g_theta_m = g_theta_m.cuda()
        self.data_num = data_num


        self.zero_layer = torch.zeros_like(self.segments_tensor)
        self.one_layer = torch.ones_like(self.segments_tensor)

    
    def __len__(self):
        return self.data_num
    
    def __getitem__(self, index):

        attack_region_tmp = self.zero_layer.clone()
        flag = torch.rand_like( self.segments_label) < self.g_theta_m        
        for i in range(flag.shape[0]):
            if flag[i]:
                sp =  self.segments_label[i]
                attack_region_tmp = torch.where(self.segments_tensor==sp, self.one_layer, attack_region_tmp)
        
        # # get convex envolope
        # attack_region_tmp_np = attack_region_tmp.cpu().numpy()
        # attack_region_tmp_label_np = measure.label(attack_region_tmp_np)
        # connect_region_number = int(np.max(attack_region_tmp_label_np))
        
        # one_np = np.ones_like(attack_region_tmp_np)
        # zero_np = np.zeros_like(attack_region_tmp_np)
        # attack_region_envolope_np = np.zeros_like(attack_region_tmp_np)

        # for i in range(connect_region_number):
        #     binary_map = np.where(attack_region_tmp_label_np==i+1, one_np, zero_np)
        #     convex_env = convex_hull_image(binary_map)

        #     attack_region_envolope_np = attack_region_envolope_np + convex_env
        #     pass


        # attack_region_tmp = torch.from_numpy(attack_region_envolope_np)
        # attack_region_tmp = torch.clamp(attack_region_tmp, 0, 1).cuda()

        return attack_region_tmp, flag

if __name__=='__main__':
    segments_tensor = [
        [0,0,1,1,1,2,2,2,3,3,4,4,5,5,5,0,0],
        [0,0,1,1,1,2,2,2,3,3,4,4,4,5,5,0,0],
        [0,0,1,1,1,2,3,3,3,3,4,4,4,5,5,0,0],
        [0,0,1,1,1,2,2,2,3,3,4,4,6,6,5,0,0],
        [0,0,1,1,1,2,2,2,3,3,4,4,6,6,5,0,0],
        [0,0,1,1,1,2,2,2,3,3,4,4,6,6,5,0,0],
        [0,0,1,1,1,2,2,2,3,3,4,4,6,6,5,0,0],
        [0,0,1,1,1,2,2,2,3,3,4,4,6,6,5,0,0],
        [0,0,1,1,1,2,2,2,3,3,4,4,6,6,5,0,0],
        [0,0,1,1,1,2,2,2,3,3,4,4,6,6,5,0,0],
        [0,0,1,1,1,2,2,2,3,3,4,4,6,6,5,0,0],
        [0,0,1,1,1,2,2,2,3,3,4,4,6,6,5,0,0],
        ]
    segments_tensor = torch.Tensor(segments_tensor)
    g_theta_m = torch.Tensor([0.1,0.2,0.3,0.4,0.5,0.6])
    data_num = 555
    data = SuperPixelGet(torch.Tensor([1,2,3,4,5,6]), segments_tensor, g_theta_m, data_num)
    dataloader = DataLoader(data, batch_size=128,shuffle=False) #使用DataLoader加载数据

    max_len = 0
    for epoch in range(10):
        for i_batch, batch_data in enumerate(dataloader):

            sum_tensor = torch.sum(batch_data, dim=0)
            sum_tensor = sum_tensor/torch.max(sum_tensor)
            sum_tensor = sum_tensor.unsqueeze(0).unsqueeze(0)
            sum_tensor = F.interpolate(sum_tensor, (800, 800), mode='nearest').squeeze()
            sum_pil = transforms.ToPILImage()(sum_tensor)
            sum_pil.show()

            pass


