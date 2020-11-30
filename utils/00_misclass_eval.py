import os
import numpy as np

iou_dir = '../common_data/NES_attack/YOLO3/iou10'
iou_list = os.listdir(iou_dir)

success_list = os.listdir('../common_data/NES_attack/YOLO3/success10')
fail = 0
iou_fail = 0
for iou_file in iou_list:
    record_path = os.path.join(iou_dir, iou_file)
    with open(record_path,"r") as f:
        min_max_iou_old = f.read()
        min_max_iou_old = float(min_max_iou_old)
        if min_max_iou_old>0.5:
            iou_fail = iou_fail + 1
            name = iou_file.split('.')[0]+'.png'
            if not name in success_list:
            #fail
                fail = fail + 1
        
print(iou_fail)
print(fail)



 
success_all_list = []
for iou_file in iou_list:
    record_path = os.path.join(iou_dir, iou_file)
    with open(record_path,"r") as f:
        min_max_iou_old = f.read()
        min_max_iou_old = float(min_max_iou_old)
        if min_max_iou_old<0.5:
            success_all_list.append(iou_file.split('.')[0])

for cls_fail in success_list:
    success_all_list.append(cls_fail.split('.')[0])

success_all_np = np.array(success_all_list)
success_all_np = np.unique(success_all_np)
print(success_all_np.shape, len(iou_list))
