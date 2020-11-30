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
         self.config_file = './configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'
        self.checkpoint_file = '../common_data/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'
        self.Mask_RCNN = init_detector(self.config_file, self.checkpoint_file, device='cpu').cuda()
      
        self.mean = torch.Tensor([123.675, 116.28, 103.53]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()
        self.std = torch.Tensor([58.395, 57.12, 57.375]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()


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
        img_frcn = get_Image_ready(self.Mask_RCNN, '1016.png')
        img_frcn['img_metas'][0][0]['filename'] = None
        img_frcn['img_metas'][0][0]['ori_filename'] = None
        img_frcn['img_metas'][0][0]['ori_shape'] = None
        img_frcn['img_metas'][0][0]['pad_shape'] = None
        img_frcn['img_metas'][0][0]['scale_factor'] = None

        # attack_area_rate = 0.2
        ATTACK_AREA_RATE = 0.1
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

            # (0) prepare stack of sp
            zero_layer = torch.zeros_like(segments_tensor)
            one_layer = torch.ones_like(segments_tensor)
            segments_stack = torch.stack([torch.where(segments_tensor==segments_label[j], segments_tensor, zero_layer) for j in range(segments_label.shape[0])], dim=0)
            

            
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

           

            # attack_region_tmp = attack_region_rand
            # attack_region_tmp = attack_region_fast
            attack_region_tmp = attack_region_search_top
            # attack_region_tmp = attack_region_four_square
            # attack_region_tmp = sandian_region

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

            t_op_num = 1500
            min_max_iou_record = 1
            for t_op_step in range(t_op_num):
                adv_patch = torch.sigmoid(adv_patch_w)
                patched_img = torch.where(attack_region_tmp>0, adv_patch, img.cuda()).unsqueeze(0)
                # @@@@!!!!!
                # patched_img = torch.where(mask.cuda()>0, patched_img, mask.unsqueeze(0).repeat(3,1,1).cuda())

                patched_img_255 = patched_img * 255.
                patched_img_rsz = F.interpolate(patched_img_255, (800, 800), mode='bilinear').cuda()
                patched_img_nom_rsz = (patched_img_rsz - self.mean) / self.std


                # output
                img_new = copy.deepcopy(img_frcn)
                img_new['img'][0] = patched_img_nom_rsz
                # [0] = patched_img_batch_nom_rsz
                frcn_output = self.Mask_RCNN(return_loss=False, rescale=False,  **img_new)


                 # compute loss
                proposals_4507 = frcn_output[1]
                proposals_score_4507 = frcn_output[2]
                det_bboxes, det_labels, proposals = frcn_output[0]

                det_bboxes = torch.cat(det_bboxes,dim=0)/800*500
                proposals = torch.cat(proposals,dim=0)/800*500
                det_labels = torch.cat(det_labels,dim=0)

                
                attack_prob = det_labels[:,class_label]
                training_confidence_threshold = 0.05
                ov_thrs_index = torch.where(attack_prob > training_confidence_threshold)[0] # for certain class
                pbbox_attack_cls = det_bboxes[:, class_label*4:(class_label+1)*4]

                


                # cls loss
                attack_class_score = det_labels[:,class_label]
                top_sort_class_score = torch.sort(attack_class_score, descending=True)[0][:20]
                cls_loss = torch.sum(top_sort_class_score)


                # iou loss
                bbox_x1 = bbox[0]/500
                bbox_y1 = bbox[1]/500
                bbox_w = bbox[2]/500
                bbox_h = bbox[3]/500
                ground_truth_bbox = [bbox_x1, bbox_y1, bbox_x1+bbox_w, bbox_y1 + bbox_h]
                ground_truth_bbox = torch.Tensor(ground_truth_bbox).unsqueeze(0).cuda()
                iou_all = compute_iou_tensor(det_bboxes[:, class_label*4:(class_label+1)*4], ground_truth_bbox)
                iou_positive = iou_all[iou_all>0.05]
                iou_loss = torch.sum(iou_all)

                # class loss selected by IoU
                attack_class_score = det_labels[:,class_label]
                attack_class_score_iou = attack_class_score[iou_all>0.25]
                attack_class_score_iou_sort = torch.sort(attack_class_score_iou, descending=True)[0][:30]
                cls_iou_loss = torch.sum(attack_class_score_iou_sort)




                final_roi = pbbox_attack_cls[ov_thrs_index]  # for certain class
                final_roi = final_roi[:,:4]
                final_ctx = (final_roi[:,0] + final_roi[:,2])/2
                final_cty = (final_roi[:,1] + final_roi[:,3])/2
                final_ctx = final_ctx.unsqueeze(-1)
                final_cty = final_cty.unsqueeze(-1)
                final_roi_target = torch.cat([final_ctx,final_cty,final_ctx,final_cty], dim=-1)
                reg_loss = 10* l1_norm(final_roi-final_roi_target) / final_roi.shape[0] /500



                # RPN loss
                # r1 : from score
                # r2 : from x,y,w,h

                # rpn score target is 0
                rpn_score = proposals[:, 4]
                loss_r1 = l2_norm(rpn_score - 0)

                # rpn box target is smaller the proposal boxes
                rpn_ctx = (proposals[:,0] + proposals[:,2])/2
                rpn_cty = (proposals[:,1] + proposals[:,3])/2
                rpn_box = proposals[:,:4]
                rpn_ctx = rpn_ctx.unsqueeze(-1)
                rpn_cty = rpn_cty.unsqueeze(-1)
                rpn_box_target = torch.cat([rpn_ctx,rpn_cty,rpn_ctx,rpn_cty], dim=-1)
                # loss_r2 = l1_norm(rpn_score.unsqueeze(-1).repeat(1,4)*(rpn_box - rpn_box_target)) / 500
                loss_r2 = l1_norm((rpn_box - rpn_box_target)) / 500
                

                lambda_balance1 = 0.02
                # rpn_loss = loss_r1 + lambda_balance1 * loss_r2
                rpn_loss = lambda_balance1 * loss_r2
                # rpn_loss = loss_r1 



                total_loss = cls_loss + cls_iou_loss + reg_loss + rpn_loss 
                total_loss = cls_iou_loss + reg_loss + rpn_loss
                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()




                # ----------------------------------
                # ------------------------
                # early stop

                #test
                patched_img_cpu = patched_img.cpu().squeeze()
                test_confidence_threshold = 0.45


                
                ov_test_thrs_index = torch.where(attack_prob > test_confidence_threshold)[0]

                final_pbbox = det_bboxes[:, class_label*4:(class_label+1)*4]
                ground_truth_bboxs_final = ground_truth_bbox.repeat(final_pbbox.shape[0],1)
                iou = compute_iou_tensor(final_pbbox, ground_truth_bboxs_final)
                attack_prob_select_by_iou_ = attack_prob[iou>0.05]
                attack_prob_select_by_iou_ = attack_prob_select_by_iou_[attack_prob_select_by_iou_>test_confidence_threshold]


                # stop if no such class found
                if attack_prob_select_by_iou_.shape[0] == 0:
                    print('Break at',t_op_step,'no bbox found')
                    # save image
                    patched_img_cpu_pil = transforms.ToPILImage()(patched_img_cpu)
                    out_file_path = os.path.join('../common_data/NES_attack/success'+str(int(ATTACK_AREA_RATE*100)), img_name)
                    patched_img_cpu_pil.save(out_file_path)
                    break

                # stop judgement max same class IoU < 0.5

                # max same-class object bounding box iou s
                final_pbbox = det_bboxes[ov_test_thrs_index][:, class_label*4:(class_label+1)*4]
                ground_truth_bboxs_final = ground_truth_bbox.repeat(final_pbbox.shape[0],1)
                iou = compute_iou_tensor(final_pbbox, ground_truth_bboxs_final)
                iou_max = torch.max(iou)
                if iou_max < 0.05:
                    print('Break at',t_op_step,'iou final max:', torch.max(iou))
                    # save image
                    patched_img_cpu_pil = transforms.ToPILImage()(patched_img_cpu)
                    out_file_path = os.path.join('../common_data/NES_attack/success'+str(int(ATTACK_AREA_RATE*100)), img_name)
                    patched_img_cpu_pil.save(out_file_path)

                  
                    break

                # report 
                ground_truth_bboxs = ground_truth_bbox.repeat(1000,1)
                final_pbbox = det_bboxes[ov_test_thrs_index][:, class_label*4:(class_label+1)*4]
                ground_truth_bboxs_final = ground_truth_bbox.repeat(final_pbbox.shape[0],1)
                iou = compute_iou_tensor(final_pbbox, ground_truth_bboxs_final)
                
                max_iou = torch.max(iou)
                if max_iou < min_max_iou_record:
                    min_max_iou_record = max_iou
                    txt_save_dir =  '../common_data/NES_attack/iou'+str(int(ATTACK_AREA_RATE*100))
                    txt_save_path = os.path.join(txt_save_dir, img_name.split('.')[0]+'.txt')
                    with open(txt_save_path,'w') as f:
                        text = str(float(max_iou))
                        f.write(text)

                if t_op_step % 100 == 0:

                    iou_sort = torch.sort(iou,descending=True)[0][:6].detach().clone().cpu()

                    print(t_op_step, 'iou t-cls  :', iou_sort)

                    # iou over 0.5, confidence print
                    final_pbbox = det_bboxes[:, class_label*4:(class_label+1)*4]
                    iou = compute_iou_tensor(final_pbbox, ground_truth_bbox.repeat(final_pbbox.shape[0],1))
                    attack_prob = det_labels[:,class_label]
                    attack_prob_select_by_iou_ = attack_prob[iou>0.05]

                    attack_prob_select_by_iou_sort = torch.sort(attack_prob_select_by_iou_,descending=True)[0][:6].detach().cpu()
                    print(t_op_step, 'right cls cf:', attack_prob_select_by_iou_sort)





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


