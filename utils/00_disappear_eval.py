import os
import numpy as np

area_dir = '../common_data/one_patch_attack/disappear/YOLO3/area'
area_list = os.listdir(area_dir)

area_sum = 0
for area_file in area_list:
    record_path = os.path.join(area_dir, area_file)
    with open(record_path,"r") as f:
        area = f.read()
        area = float(area)
    area_sum = area_sum + area
print(area_sum/len(area_list))

            
      