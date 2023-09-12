import sys
import time

import numpy as np
from PIL import Image
import os

# imgs_path = '/home/asus/Documents/4T/zyq/datasets/Depth/nyu/training'
# imgs_path_list = os.listdir(imgs_path)
# count = 0
#
# img_narray = []
# for img_path_list in imgs_path_list:
#         imgs_depth = imgs_path + '/' + img_path_list + '/leres_depth'
#         if os.path.isdir(imgs_depth):
#             imgs_depth_list = os.listdir(imgs_depth)
#             for img_depth in imgs_depth_list:
#                 img1 = Image.open(imgs_depth + '/' + img_depth)
#                 max = np.unique(np.array(img1))
#                 if max[-1] == 255:
#                     if len(max) > 1:
#                         img_narray.append(max[-2])
#                 else:
#                     img_narray.append(max[-1])
#                 count = count + 1
#                 sys.stdout.write('\r finish{0}'.format(count))
#                 sys.stdout.flush()
# print('__')
# print(np.max(img_narray))

# imgs_path = '/home/asus/Documents/4T/zyq/Download/datasets/coco/panoptic_train2017'
# imgs_path_list = os.listdir(imgs_path)
# count = 0
#
# img_narray = []
# for img_path_list in imgs_path_list:
#         img1 = Image.open(imgs_path + '/' + img_path_list)
#         max = np.unique(np.array(img1))
#         if max[-1] == 255:
#             if len(max) > 1:
#                 img_narray.append(max[-2])
#         else:
#             img_narray.append(max[-1])
#         count = count + 1
#         sys.stdout.write('\r finish{0}'.format(count))
#         sys.stdout.flush()
# print('__')
# print(np.max(img_narray))

img_path = '/home/xibi/Desktop/2dmask_results/scene0000_00/0.png'
img = Image.open(img_path)
img_array = np.unique(np.array(img))
print('kkk')