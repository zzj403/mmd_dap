"""
Training code for Adversarial patch training using Faster RCNN based on mmdetection
Redo UPC in PyTorch

"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

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
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel



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
from train_patch_frcn_measure_no_stop_gray_step_3_0920 import measure_region_with_attack
import math
from dataset.coco_train_1000_PERSON import CocoTrainPerson
from dataset.sp_geter_dataset import SuperPixelGet

from torch.utils.data import DataLoader,Dataset
from utils.tv_loss import TVLoss
from utils.iou import compute_iou_tensor

seed = 2020
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

csv_name = 'x_result2.csv'
# torch.cuda.set_device(0)

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
        # self.config = patch_config.patch_configs[mode]()

        # self.darknet_model = Darknet(self.config.cfgfile)
        # self.darknet_model.load_weights(self.config.weightfile)
        # self.darknet_model = self.darknet_model.eval().cuda() # TODO: Why eval?

        self.ocpy_list = []
        for i in range(4):
            if i ==0:
                ocpy_tensor = torch.rand(500,500,500,3).cuda(i)
                self.ocpy_list.append(ocpy_tensor)
                continue

            ocpy_tensor = torch.rand(500,500,500,6).cuda(i)
            self.ocpy_list.append(ocpy_tensor)
            
        
        self.config_file = './configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
        self.checkpoint_file = '../common_data/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
        self.Faster_RCNN = init_detector(self.config_file, self.checkpoint_file, device='cpu').cuda()

        # self.Faster_RCNN = MMDistributedDataParallel(
        #     self.Faster_RCNN.cuda(),
        #     device_ids=[torch.cuda.current_device()],
        #     broadcast_buffers=False)

        self.Faster_RCNN = torch.nn.DataParallel(self.Faster_RCNN)
        
        # self.patch_applier = PatchApplier().cuda()
        # self.patch_transformer = PatchTransformer().cuda()
        # self.prob_extractor = MaxProbExtractor(0, 80, self.config).cuda()
        # self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size).cuda()
        # self.total_variation = TotalVariation().cuda()
        self.mean = torch.Tensor([123.675, 116.28, 103.53]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()
        self.std = torch.Tensor([58.395, 57.12, 57.375]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()

        # self.writer = self.init_tensorboard(mode)


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
        n_epochs = 5000
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
        
        data_obj = CocoTrainPerson(dataType='train2017',num_use=500)
        dataloader_obj = DataLoader(data_obj, batch_size=1,shuffle=False) #使用DataLoader加载数据

        # img info prepare
        img_frcn = get_Image_ready(self.Faster_RCNN.module, '1016.png')
        img_frcn['img_metas'][0][0]['filename'] = None
        img_frcn['img_metas'][0][0]['ori_filename'] = None
        img_frcn['img_metas'][0][0]['ori_shape'] = None
        img_frcn['img_metas'][0][0]['pad_shape'] = None
        img_frcn['img_metas'][0][0]['scale_factor'] = None

        # attack_area_rate = 0.2
        ATTACK_AREA_RATE = 0.1
        evo_step_num = 20
        batch_size_sp = 12
        population_num = 72 # 36
        optim_step_num = 300
        k = 0
        for i_batch, batch_data in enumerate(dataloader_obj):
            img, mask, bbox, class_label = batch_data[0][0], batch_data[1][0], batch_data[2][0], batch_data[3][0]
            # img  : 3,500,500
            # mask : 500,500
            # bbox : x1,y1,w,h
            # class_label : tensor[]

            img_name = batch_data[4][0]

            print('---------------')
            print(img_name)
            print('---------------')

            save_dir = os.path.join('../common_data/NES_search_test_1107/', img_name.split('.')[0])
            if os.path.exists(save_dir):
                continue
            time0 = time.time()



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
            segments_stack = segments_stack.cpu()

            mask_area = torch.sum(mask)
            # (1) find the center sp
            bbox_x1 = bbox[0]
            bbox_y1 = bbox[1]
            bbox_w = bbox[2]
            bbox_h = bbox[3]

            bbox_x_c = bbox_x1 + bbox_w/2
            bbox_y_c = bbox_y1 + bbox_h/2
            bbox_x_c_int = int(bbox_x_c)
            bbox_y_c_int = int(bbox_y_c)

            # if segments_tensor[bbox_y_c_int, bbox_x_c_int] == 0:
            #     # no sp in center
            #     center_sp = torch.Tensor().cuda()
            # else:
            #     center_sp = segments_tensor[bbox_y_c_int, bbox_x_c_int].unsqueeze(0)
            #     center_sp_layer = torch.where(segments_tensor==center_sp, one_layer, zero_layer)
            #     center_sp_layer_np = center_sp_layer.cpu().numpy()
            #     kernel = np.ones((3,3),np.uint8)  
            #     center_sp_layer_dilate = cv2.dilate(center_sp_layer_np, kernel, iterations = 2)
            #     center_sp_layer_dilate = torch.from_numpy(center_sp_layer_dilate)
            #     center_sp_layer_dilate_stack = center_sp_layer_dilate.unsqueeze(0).repeat(segments_stack.shape[0],1,1)

            #     cross_stack = center_sp_layer_dilate_stack * segments_stack
            #     cross_ = torch.sum(cross_stack, dim=0)
            #     neighborhoods = torch.unique(cross_)[1:]
            #     center_sps = neighborhoods
            #     cross_stack = cross_stack.cpu()
            #     center_sp_layer_dilate_stack = center_sp_layer_dilate_stack.cpu()



            #     # we also need center sp's neighborhoods




            # # (2) find the boundary sp

            # # boundary_erode
            # kernel = np.ones((3,3),np.uint8)
            # mask_erosion = cv2.erode(mask_np, kernel, iterations = 2)
            # boundary_erode = mask_np - mask_erosion
            # boundary_erode = torch.from_numpy(boundary_erode)
            # # boundary_erode_pil = transforms.ToPILImage()(boundary_erode.cpu())
            # # boundary_erode_pil.show()

            # boundary_erode_stack = boundary_erode.unsqueeze(0).repeat(segments_stack.shape[0],1,1)
            # boundary_mul_segments_stack = boundary_erode_stack * segments_stack
            # boundary_mul_segments = torch.sum(boundary_mul_segments_stack, dim=0)
            # # boundary_mul_segments_pil = transforms.ToPILImage()(boundary_mul_segments.cpu())
            # # boundary_mul_segments_pil.show()
            # boundary_mul_segments_unique = torch.unique(boundary_mul_segments)
            # boundary_mul_segments_unique = boundary_mul_segments_unique[1:]
            # boundary_sp = boundary_mul_segments_unique




            # shan dian
            # init grid
            densy = 7
            unit_w = 13*densy
            unit_h = 13*densy
            sandian = torch.zeros(unit_w,unit_h)
            '''
            log:
            10,5,10,5 : 0.04   work! at 700
            10,5,10,6 : 0.0333 work! at 2040
            '''
            sandian = sandian.reshape(13,densy,13,densy)
            sandian[:,int((densy-1)/2),:,int((densy-1)/2)] = 1
            sandian = sandian.reshape(unit_w, unit_h)
            sandian = sandian.unsqueeze(0).unsqueeze(0)
            sandian = F.interpolate(sandian, (500, 500), mode='nearest').squeeze()
            sandian_stack = sandian.unsqueeze(0).repeat(segments_stack.shape[0],1,1)
            sandian_mul_segments_stack = sandian_stack * segments_stack
            sandian_mul_segments = torch.sum(sandian_mul_segments_stack, dim=0)
            sandian_mul_segments_pil = transforms.ToPILImage()(sandian_mul_segments.cpu())
            sandian_mul_segments_pil.show()
            sandian_mul_segments_unique = torch.unique(sandian_mul_segments)
            sandian_mul_segments_unique = sandian_mul_segments_unique[1:]
            sandian_sp = sandian_mul_segments_unique




            # pay attention
            # spot_sp = torch.cat((center_sps, boundary_sp), dim=0)
            spot_sp = sandian_sp
            spot_sp = torch.unique(spot_sp).long()


            sandian_stack = sandian_stack.cpu()
            # boundary_erode_stack = boundary_erode_stack.cpu()
            # boundary_mul_segments_stack = boundary_mul_segments_stack.cpu()
            segments_stack = segments_stack.cpu()
            
            
            torch.cuda.empty_cache()
            
            # show_tensor = img.clone().cuda()
            # for i in range(spot_sp.shape[0]):
            #     sp = spot_sp[i]
            #     show_tensor = torch.where(segments_tensor==sp, zero_layer.fill_(1).unsqueeze(0).repeat(3,1,1),show_tensor)
            
            # show_tensor_pil = transforms.ToPILImage()(show_tensor.cpu())
            # show_tensor_pil.show()


            # generate theta_m
            # for sp id from 1 to 128
            uniform_ratio = torch.Tensor([ATTACK_AREA_RATE])[0]
            higher_ratio =  uniform_ratio*1.5
            uniform_ratio_theta = 1/2*torch.log(uniform_ratio/(1-uniform_ratio))
            higher_ratio_theta = 1/2*torch.log(higher_ratio/(1-higher_ratio))
            theta_m = torch.zeros_like(segments_label).cpu().fill_(uniform_ratio_theta)
            theta_m[spot_sp-1] = higher_ratio_theta



            
            for evo_step in range(evo_step_num):
                # prepare sp dataset
                g_theta_m = 1/2*(torch.tanh(theta_m)+1)
                data_sp = SuperPixelGet(segments_label = segments_label, 
                                    segments_tensor=segments_tensor, 
                                    g_theta_m=g_theta_m,
                                    data_num=population_num)
                dataloader_sp = DataLoader(data_sp, batch_size=batch_size_sp,shuffle=False) #使用DataLoader加载数据

                F_value_restore = torch.Tensor()
                select_m_restore = torch.Tensor()

                for j_sp_batch , sp_batch_data in enumerate(dataloader_sp):

                    attack_mask_batch, select_m = sp_batch_data[0].unsqueeze(1), sp_batch_data[1]
                    select_m_restore = torch.cat((select_m_restore, select_m.cpu()))
                    
                    batch_size_now = attack_mask_batch.shape[0]
                    ## start at gray
                    adv_patch_w_batch = torch.zeros(batch_size_now,3,500,500).cuda()

                    adv_patch_w_batch.requires_grad_(True)

                    ## optimizer and scheduler
                    # optimizer = optim.Adam([
                    #     {'params': adv_patch, 'lr': 0.01*255}
                    # ], amsgrad=True)
                    optimizer = optim.Adam([
                        {'params': adv_patch_w_batch, 'lr': 0.1}
                    ], amsgrad=True)

                    L_value_step = torch.ones(batch_size_now ,optim_step_num)*1000
                    for step in tqdm(range(optim_step_num)):
                        
                        # prepare batch data to feed the frcn
                        img_batch = img.cuda().unsqueeze(0).repeat(batch_size_now,1,1,1)
                        adv_patch_batch = torch.sigmoid(adv_patch_w_batch/2)
                        patched_img_batch = torch.where(attack_mask_batch > 0, adv_patch_batch, img_batch)
                        patched_img_batch_255 = patched_img_batch * 255.
                        patched_img_batch_rsz = F.interpolate(patched_img_batch_255, (800, 800), mode='bilinear').cuda()
                        patched_img_batch_nom_rsz = (patched_img_batch_rsz - self.mean) / self.std


                        # output
                        img_new = copy.deepcopy(img_frcn)
                        img_new['img'][0] = patched_img_batch_nom_rsz
                        # [0] = patched_img_batch_nom_rsz
                        img_new['img_metas'][0] = [img_new['img_metas'][0][0] for i in range(batch_size_now)]
                        frcn_output = self.Faster_RCNN(return_loss=False, rescale=False,  **img_new)
                        # output formate is [x1,y1,x2,y2]


                        # compute loss
                        proposals_4507 = frcn_output[1]
                        proposals_score_4507 = frcn_output[2]
                        det_bboxes, det_labels, proposals = frcn_output[0]

                        det_bboxes = torch.cat(det_bboxes,dim=0)/800*500
                        proposals = torch.cat(proposals,dim=0)/800*500
                        det_labels = torch.cat(det_labels,dim=0)

                        
                        attack_prob = det_labels[:,class_label]
                        training_confidence_threshold = 0.28
                        ov_thrs_index = torch.where(attack_prob > training_confidence_threshold)[0] # for certain class
                        pbbox_attack_cls = det_bboxes[:, class_label*4:(class_label+1)*4]

                        


                        # cls loss
                        attack_class_score = det_labels[:,class_label]
                        top_sort_class_score = torch.sort(attack_class_score, descending=True)[0][:10]
                        cls_loss = torch.sum(top_sort_class_score)

                        # iou loss
                        bbox_x1 = bbox[0]/500*416
                        bbox_y1 = bbox[1]/500*416
                        bbox_w = bbox[2]/500*416
                        bbox_h = bbox[3]/500*416
                        ground_truth_bbox = [bbox_x1, bbox_y1, bbox_x1+bbox_w, bbox_y1 + bbox_h]
                        ground_truth_bbox = torch.Tensor(ground_truth_bbox).unsqueeze(0).cuda()
                        iou_all = compute_iou_tensor(det_bboxes[:, class_label*4:(class_label+1)*4], ground_truth_bbox)
                        iou_positive = iou_all[iou_all>0.15]
                        iou_loss = torch.sum(iou_all)

                        # class loss selected by IoU
                        attack_class_score = det_labels[:,class_label]
                        attack_class_score_iou = attack_class_score[iou_all>0.45]
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

                        # rpn box target is smaller the boxes
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



                        total_loss = cls_loss + reg_loss + rpn_loss
                        total_loss = cls_loss + cls_iou_loss + reg_loss + rpn_loss

                        # if epoch > 500:
                        #     total_loss = rpn_loss + cls_loss_new + reg_loss

                        total_loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()



                        # compute F(m,t*;y)

                        # L in disappearing 
                        # i think we better use iou>0.45 in prediction, all bbox's person class confidence sum

                        bbox_x1 = bbox[0]
                        bbox_y1 = bbox[1]
                        bbox_w = bbox[2]
                        bbox_h = bbox[3]
                        ground_truth_bbox = [bbox_x1, bbox_y1, bbox_x1+bbox_w, bbox_y1 + bbox_h]
                        ground_truth_bbox = torch.Tensor(ground_truth_bbox).unsqueeze(0).cuda()
                        iou_all = compute_iou_tensor(det_bboxes[:, class_label*4:(class_label+1)*4], ground_truth_bbox)

                        iou_select_index = iou_all > 0.45

                        
                        for ib in range(batch_size_now):

                            det_labels_one = det_labels[ib:(ib+1)*1000, class_label]

                            iou_select_index_one = iou_select_index[ib:(ib+1)*1000]

                            cls_conf_iou_select = det_labels_one[iou_select_index_one]

                            cls_conf_iou_select = cls_conf_iou_select[cls_conf_iou_select>0.25]

                            cls_conf_iou_select_top = torch.sort(cls_conf_iou_select, descending=True)[0][:10]

                            L_value_step[ib, step] = torch.sum(cls_conf_iou_select_top).detach().clone()
                            pass
                        if (torch.min(L_value_step, dim=1)[0] == torch.zeros(batch_size_now)).all():
                            break
                        

                        pass
                    L_value = - torch.min(L_value_step, dim=1)[0]
                    F_value = L_value - torch.sum(torch.sum(sp_batch_data[0], dim=-1), dim=-1).cpu()/mask_area*10 ###!!!!
                    F_value_restore = torch.cat((F_value_restore, F_value))

                    
                    pass
                # print(F_value_restore)
                # now we have F value 

                delta_J_theta = 1/population_num * F_value_restore.unsqueeze(1) * 2 * (select_m_restore.cpu().float() - g_theta_m)
                
                delta_J_theta = torch.sum(delta_J_theta, dim=0)
                theta_m = theta_m + delta_J_theta
                g_theta_m = 1/2*(torch.tanh(theta_m)+1)

                select_sp_index = torch.sort(theta_m)[1][:int(theta_m.shape[0]*ATTACK_AREA_RATE)]

                attack_region_show = zero_layer.clone().squeeze().cuda()
                flag = torch.zeros_like(segments_label) 
                flag[select_sp_index] = 1
                for i in range(segments_label.shape[0]):

                    color = g_theta_m[i]
                    color_layer = torch.zeros_like(segments_tensor).fill_(color)
                    sp =  segments_label[i]

                    attack_region_show = torch.where(segments_tensor==sp, color_layer, attack_region_show)
                attack_region_show = attack_region_show / torch.max(attack_region_show)

                attack_region_show = (attack_region_show + img.cuda())/2
                
                attack_region_show_pil = transforms.ToPILImage()(attack_region_show.cpu())

                save_dir = os.path.join('../common_data/NES_search_test_1107/', img_name.split('.')[0])
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = os.path.join(save_dir, str(evo_step)+'_pop_'+str(population_num)+'_'+str(base_SLIC_seed_num)+'.png')
                attack_region_show_pil.save(save_path)

                time1 = time.time()

                time_cost = time1 - time0

                if time_cost > 1800:
                    break
                
                print(g_theta_m)
        print(assasasasasasa)
        pass











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


