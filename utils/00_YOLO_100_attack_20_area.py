"""
Training code for Adversarial patch training using Faster RCNN based on mmdetection
Redo UPC in PyTorch
sp_lr = 0.3
pop = 300
random texture
evo_step_num = 40
"""

import PIL
import cv2
# from load_data import *
import copy
from tqdm import tqdm
from mmdet import __version__
from mmdet.apis import init_detector, inference_detector, show_result_pyplot, get_Image_ready
import mmcv
from mmcv.ops import RoIAlign, RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

import torch
import gc
import matplotlib.pyplot as plt
from torch import autograd
from torchvision import transforms

import subprocess
# from utils.utils import *
import numpy as np
import pickle

# import patch_config as patch_config
import sys
import time

# from brambox.io.parser.annotation import DarknetParser as anno_darknet_parse
# from utils import *
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import io
from skimage.segmentation import mark_boundaries
import os
from PIL import Image
import torch.optim as optim
import torch.nn.functional as F
import skimage.io as io  
# from train_patch_frcn_measure_no_stop_gray_step_3_0920 import measure_region_with_attack
import math
from dataset.coco_train_1000_PERSON import CocoTrainPerson
from dataset.sp_geter_dataset import SuperPixelGet

from torch.utils.data import DataLoader,Dataset
from utils.tv_loss import TVLoss
from utils.iou import compute_iou_tensor
from get_convex_env import get_conv_envl


seed = 2020
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

csv_name = 'x_result2.csv'
torch.cuda.set_device(0)

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    import scipy.stats as st

    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel



class PatchTrainer(object):
    def __init__(self, mode):

        self.config_file = './configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
        self.checkpoint_file = '../common_data/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
        self.Faster_RCNN = init_detector(self.config_file, self.checkpoint_file, device='cpu').cuda()

        self.yolo_config_file = './configs/yolo/yolov3_d53_mstrain-416_273e_coco.py'
        self.yolo_checkpoint_file = '../common_data/yolov3_d53_mstrain-416_273e_coco-2b60fcd9.pth'
        self.YOLOv3 = init_detector(self.yolo_config_file, self.yolo_checkpoint_file, device='cpu').cuda()

        self.mean = torch.Tensor([0,  0,  0]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()
        self.std = torch.Tensor([255., 255., 255.]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()



    def Transform_Patch(self, patch):
        # clamp_patch = torch.clamp(patch, 0.01, 254.99)
        clamp_patch = patch
        unsqueezed_patch = clamp_patch.unsqueeze(0)
        resized_patch = F.interpolate(unsqueezed_patch, (800, 800), mode='bilinear').cuda()
        normalized_patch = (resized_patch - self.mean) / self.std
        return normalized_patch
    
    def Transform_Patch_batch(self, patch):
        clamp_patch = torch.clamp(patch, 0.01, 254.99)
        resized_patch = F.interpolate(clamp_patch, (800, 800), mode='bilinear').cuda()
        normalized_patch = (resized_patch - self.mean) / self.std
        return normalized_patch

    def train(self):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """

        img_size = 800
        batch_size = 1
        n_t_op_steps = 5000
        max_lab = 14

        ATTACK_TASK = 'target'

        # TARGET_CLASS = 'dog'
        TARGET_CLASS = 16
        # ATTACK_TASK = 'untarget'

        time_str = time.strftime("%Y%m%d-%H%M%S")



        conv_size = 2
        kernel1 = gkern(2*conv_size+1, 3).astype(np.float32)
        stack_kernel1 = np.stack([kernel1, kernel1, kernel1]).swapaxes(2, 0)
        stack_kernel1 = np.expand_dims(stack_kernel1, 3)
        stack_kernel1 = torch.from_numpy(stack_kernel1).permute(2,3,0,1).float()



        # Dataset prepare
        
        data_obj = CocoTrainPerson(dataType='train2017',num_use=100)

        dataloader_obj = DataLoader(data_obj, batch_size=1,shuffle=False) #使用DataLoader加载数据

        # img info prepare
        img_frcn = get_Image_ready(self.Faster_RCNN, '1016.png')
        img_frcn['img_metas'][0][0]['filename'] = None
        img_frcn['img_metas'][0][0]['ori_filename'] = None
        img_frcn['img_metas'][0][0]['ori_shape'] = None
        img_frcn['img_metas'][0][0]['pad_shape'] = None
        img_frcn['img_metas'][0][0]['scale_factor'] = None

        # attack_area_rate = 0.2
        ATTACK_AREA_RATE = 0.2
        decay_t_op_step = 100
        batch_size_sp = 3
        population_num = 300 # 36
        optim_step_num = 300
        k = 0
        for i_batch, batch_data in enumerate(dataloader_obj):
            img, mask, bbox, class_label = batch_data[0][0], batch_data[1][0], batch_data[2][0], batch_data[3][0]
            # img  : 3,500,500
            # mask : 500,500
            # bbox : x1,y1,w,h
            # class_label : tensor[]

            img_name = batch_data[4][0]
            mask_area = torch.sum(mask)
            # if img_name.split('_')[0] != '000000001815':
            #     continue

            print('---------------')
            print(img_name)
            print('---------------')


            
            # use segment SLIC
            base_SLIC_seed_num = 3000
            img_np = img.numpy().transpose(1,2,0)
            mask_np = mask.numpy()
            numSegments = int(base_SLIC_seed_num/(500*500)*torch.sum(mask))
            segments_np = slic(image=img_np, n_segments=numSegments, sigma=0, slic_zero=True, mask=mask_np)
            segments_tensor = torch.from_numpy(segments_np).float().cuda()
            segments_label = torch.unique(segments_tensor)
            segments_label = segments_label[1:]


            # define theta_m
            # pay attention to the center and the boundary

            # (0) prepare stack of sp
            # (1) find the center sp
            # (2) find the boundary sp

            # # (0) prepare stack of sp
            zero_layer = torch.zeros_like(segments_tensor)
            one_layer = torch.ones_like(segments_tensor)
            # segments_stack = torch.stack([torch.where(segments_tensor==segments_label[j], segments_tensor, zero_layer) for j in range(segments_label.shape[0])], dim=0)
            

            
            # # (1) find the center sp
            bbox_x1 = bbox[0]
            bbox_y1 = bbox[1]
            bbox_w = bbox[2]
            bbox_h = bbox[3]

            bbox_x_c = bbox_x1 + bbox_w/2
            bbox_y_c = bbox_y1 + bbox_h/2
            bbox_x_c_int = int(bbox_x_c)
            bbox_y_c_int = int(bbox_y_c)

         

            # 3 load attack region 
            load_patch_dir = '../common_data/NES_search_test_1107/'+img_name.split('_')[0]

            load_patch_list = os.listdir(load_patch_dir)
            load_patch_list.sort()
            wat_num_max = 0
            for i_name in load_patch_list:
                wat_num = int(i_name.split('_')[0])
                if wat_num > wat_num_max:
                    wat_num_max = wat_num
            for i_name in load_patch_list:
                wat_num = int(i_name.split('_')[0])
                if wat_num == wat_num_max:
                    max_name = i_name
                    break

            load_patch = os.path.join(load_patch_dir, max_name)

            load_img = Image.open(load_patch).convert('RGB')
            load_img = transforms.ToTensor()(load_img)
            region_mask = 2*load_img - img.cpu()
            region_mask = torch.sum(region_mask,dim=0)/3
            region_mask = torch.where(mask>0, region_mask,torch.zeros_like(region_mask))


            attack_region_tmp_pil = transforms.ToPILImage()(region_mask.cpu())
            attack_region_tmp_pil.save('013k.png')
            # process mask
            region_mask_new = torch.zeros_like(region_mask).cuda()
            for i in range(segments_label.shape[0]):
                sp =  segments_label[i]
                right_color = (torch.where(segments_tensor==sp,region_mask.cuda(),one_layer*(-10))).cpu()
                right_color = torch.mean(right_color[right_color!=-10])
                color_layer = torch.ones_like(segments_tensor).fill_(right_color)
                region_mask_new = torch.where(segments_tensor==sp, color_layer, region_mask_new)      
            region_mask_new = region_mask_new
            region_mask = region_mask_new
            region_mask_unique = torch.unique(region_mask)
            for i in range(region_mask_unique.shape[0]):
                thres = region_mask_unique[i]
                # region_mask_tmp = torch.zeros_like(region_mask)
                region_mask_tmp = torch.where(region_mask>thres, one_layer, zero_layer)
                pixel_num = torch.sum(region_mask_tmp)
                if pixel_num < mask_area * ATTACK_AREA_RATE:
                    break
            attack_region_search_top = region_mask_tmp
            attack_region_search_top = get_conv_envl(attack_region_search_top)

           
            attack_region_tmp = attack_region_search_top

            attack_region_tmp = attack_region_tmp.cuda()
            print('---------------')
            print('You have used ', float(torch.sum(attack_region_tmp)/mask_area), 'area.')
            print('---------------')
             ## start at gray
            adv_patch_w = torch.zeros(3,500,500).cuda()

            adv_patch_w.requires_grad_(True)

            optimizer = optim.Adam([
                {'params': adv_patch_w, 'lr': 0.1}
            ], amsgrad=True)

            t_op_num = 800
            min_max_iou_record = 1
            for t_op_step in range(t_op_num):
                adv_patch = torch.sigmoid(adv_patch_w)
                patched_img = torch.where(attack_region_tmp>0, adv_patch, img.cuda()).unsqueeze(0)
              
                patched_img_255 = patched_img * 255.
                patched_img_rsz = F.interpolate(patched_img_255, (416, 416), mode='bilinear').cuda()
                patched_img_nom_rsz = (patched_img_rsz - self.mean) / self.std

                batch_size_now = patched_img_255.shape[0]

                # output
                img_new = copy.deepcopy(img_frcn)
                img_new['img'][0] = patched_img_nom_rsz
                yolo_output = self.YOLOv3(return_loss=False, rescale=False,  **img_new)
                # output formate is [x1,y1,x2,y2]


                # anaylize yolo_output [batch_size]
                # [
                # ( multi_lvl_bboxes, multi_lvl_cls_scores, multi_lvl_conf_scores )
                # multi_lvl_bboxes  [ 3 layers ]
                # [ [0]       1875, 4           
                #   [1]       7500, 4           
                #   [2]       30000,4   ]
                #                     
                # multi_lvl_cls_scores                    
                # [ [0]       1875, 80           
                #   [1]       7500, 80          
                #   [2]       30000,80  ]
                #                     
                # multi_lvl_conf_scores                    
                # [ [0]       1875          
                #   [1]       7500          
                #   [2]       30000     ]
                #  * batch_size
                # ]                   

                # merge yolo output
                multi_lvl_bboxes_batch = []  
                multi_lvl_cls_scores_batch = []
                multi_lvl_conf_scores_batch = []

                for i_b in range(batch_size_now):
                    multi_lvl_bboxes_batch += yolo_output[i_b][0]
                    multi_lvl_cls_scores_batch += yolo_output[i_b][1]
                    multi_lvl_conf_scores_batch += yolo_output[i_b][2]

                multi_lvl_bboxes_batch = torch.cat(multi_lvl_bboxes_batch, dim=0) 
                multi_lvl_cls_scores_batch = torch.cat(multi_lvl_cls_scores_batch, dim=0) 
                multi_lvl_conf_scores_batch = torch.cat(multi_lvl_conf_scores_batch, dim=0) 

                # objectness loss
                objectness_loss = torch.sum(multi_lvl_conf_scores_batch[multi_lvl_conf_scores_batch>0.05])

                # class loss
                attack_class_score = multi_lvl_cls_scores_batch[:,class_label]
                # attack_class_score = attack_class_score[attack_class_score>0.5]
                attack_class_score = torch.sort(attack_class_score, descending=True)[0][:30]
                cls_loss = torch.sum(attack_class_score)

                # target class loss
                attack_class_score_target = multi_lvl_cls_scores_batch[:,16]
                attack_class_score_target = attack_class_score_target[multi_lvl_conf_scores_batch>0.5]
                attack_class_score_target = attack_class_score_target[attack_class_score_target<0.9]
                attack_class_score_target = torch.sort(attack_class_score_target, descending=True)[0][:30]
                cls_target_loss = - torch.sum(attack_class_score_target)




                # iou loss
                bbox_x1 = bbox[0]/500*416
                bbox_y1 = bbox[1]/500*416
                bbox_w = bbox[2]/500*416
                bbox_h = bbox[3]/500*416
                ground_truth_bbox = [bbox_x1, bbox_y1, bbox_x1+bbox_w, bbox_y1 + bbox_h]
                ground_truth_bbox = torch.Tensor(ground_truth_bbox).unsqueeze(0).cuda()
                iou_all = compute_iou_tensor(multi_lvl_bboxes_batch, ground_truth_bbox)
                iou_positive = iou_all[iou_all>0.05]
                iou_loss = torch.sum(iou_all)


                # class loss selected by IoU
                attack_class_score = multi_lvl_cls_scores_batch[:,class_label]
                attack_class_score_iou = attack_class_score[iou_all>0.05]
                attack_class_score_iou_sort = torch.sort(attack_class_score_iou, descending=True)[0][:30]
                cls_iou_loss = torch.sum(attack_class_score_iou_sort)




                # rpn loss
                # : to make every proposal smaller to its center
                rpn_ctx = (multi_lvl_bboxes_batch[:,0] + multi_lvl_bboxes_batch[:,2])/2
                rpn_cty = (multi_lvl_bboxes_batch[:,1] + multi_lvl_bboxes_batch[:,3])/2
                rpn_box = multi_lvl_bboxes_batch[:,:4]
                rpn_ctx = rpn_ctx.unsqueeze(-1)
                rpn_cty = rpn_cty.unsqueeze(-1)
                rpn_box_target = torch.cat([rpn_ctx,rpn_cty,rpn_ctx,rpn_cty], dim=-1)
                rpn_loss = l1_norm(multi_lvl_conf_scores_batch.unsqueeze(-1).repeat(1,4)*(multi_lvl_bboxes_batch - rpn_box_target))
                


                
                # total_loss = cls_loss + objectness_loss + rpn_loss + cls_target_loss + cls_iou_loss
                # total_loss =  cls_target_loss*100 + cls_iou_loss*100  #+ rpn_loss
                total_loss =  cls_iou_loss*100  + rpn_loss

                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                



                # ----------------------------------
                # ------------------------
                # early stop
                if t_op_step %30:
                    print(  t_op_step,
                                'iou', float(torch.max(iou_all)), 
                                'cls', float(torch.max(attack_class_score)),
                                'obj', float(torch.max(multi_lvl_conf_scores_batch)))

                #test
                patched_img_cpu = patched_img.cpu().squeeze()
                test_confidence_threshold = 0.45


                iou_max = torch.max(iou_all)
                if iou_max < 0.05 or torch.max(multi_lvl_conf_scores_batch) < 0.45:
                    print('Break at',t_op_step,'iou final max:', torch.max(iou_all))
                    # save image
                    patched_img_cpu_pil = transforms.ToPILImage()(patched_img_cpu)
                    out_file_path = os.path.join('../common_data/NES_attack/YOLO3/success'+str(int(ATTACK_AREA_RATE*100)), img_name)
                    patched_img_cpu_pil.save(out_file_path)

                    
                    break

                # report 

                
                max_iou = torch.max(iou_all)
                if max_iou < min_max_iou_record:
                    min_max_iou_record = max_iou
                    txt_save_dir =  '../common_data/NES_attack/YOLO3/iou'+str(int(ATTACK_AREA_RATE*100))
                    txt_save_path = os.path.join(txt_save_dir, img_name.split('.')[0]+'.txt')
                    with open(txt_save_path,'w') as f:
                        text = str(float(max_iou))
                        f.write(text)

                if t_op_step % 100 == 0:

                    iou_sort = torch.sort(iou_all,descending=True)[0][:6].detach().clone().cpu()

                    print(t_op_step, 'iou t-cls  :', max_iou)

                  print()





        # ------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------



    def generate_patch(self, type):
        """
        Generate a random patch as a starting point for optimization.

        :param type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
        :return:
        """
        if type == 'gray':
            adv_patch_cpu = torch.full((3, 500, 500), 0.5)
        elif type == 'random':
            adv_patch_cpu = torch.rand((3, 500, 500))
        if type == 'trained_patch':
            patchfile = 'patches/object_score.png'
            patch_img = Image.open(patchfile).convert('RGB')
            patch_size = self.config.patch_size
            tf = transforms.Resize((patch_size, patch_size))
            patch_img = tf(patch_img)
            tf = transforms.ToTensor()
            adv_patch_cpu = tf(patch_img)

        return adv_patch_cpu

    def read_image(self, path):
        """
        Read an input image to be used as a patch

        :param path: Path to the image to be read.
        :return: Returns the transformed patch as a pytorch Tensor.
        """
        patch_img = Image.open(path).convert('RGB')
        tf = transforms.Resize((self.config.patch_size, self.config.patch_size))
        patch_img = tf(patch_img)
        tf = transforms.ToTensor()

        adv_patch_cpu = tf(patch_img)
        return adv_patch_cpu





def connected_domin_detect(input_img):
    from skimage import measure
    # detection
    if input_img.shape[0] == 3:
        input_img_new = (input_img[0] + input_img[1] + input_img[2])
    else:
        input_img_new = input_img
    ones = torch.Tensor(input_img_new.size()).fill_(1)
    zeros = torch.Tensor(input_img_new.size()).fill_(0)
    input_map_new = torch.where((input_img_new != 0), ones, zeros)
    # img = transforms.ToPILImage()(input_map_new.detach().cpu())
    # img.show()
    input_map_new = input_map_new.cpu()
    labels = measure.label(input_map_new[:, :], background=0, connectivity=2)
    label_max_number = np.max(labels)
    return float(label_max_number)


def get_obj_min_score(boxes):
    if type(boxes[0][0]) is list:
        min_score_list = []
        for i in range(len(boxes)):
            score_list = []
            for j in range(len(boxes[i])):
                score_list.append(boxes[i][j][4])
            min_score_list.append(min(score_list))
        return np.array(min_score_list)
    else:
        score_list = []
        for j in range(len(boxes)):
            score_list.append(boxes[j][4])
        return np.array(min(score_list))

def l2_norm(tensor):
    return torch.sqrt(torch.sum(torch.pow(tensor,2)))

def l1_norm(tensor):
    return torch.sum(torch.abs(tensor))


def main():


    trainer = PatchTrainer('paper_obj')
    trainer.train()


if __name__ == '__main__':
    main()


