# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.io as sio
# import h5py
# import os
#
# f = h5py.File("/home/asus/Documents/4T/zyq/Download/nyu_depth_v2_labeled.mat")
# images = f["images"]
# images = np.array(images)
#
# path_converted = './nyu_images'
# if not os.path.isdir(path_converted):
#     os.makedirs(path_converted)
#
# from PIL import Image
#
# images_number = []
# for i in range(len(images)):
#     images_number.append(images[i])
#     a = np.array(images_number[i])
#     r = Image.fromarray(a[0]).convert('L')
#     g = Image.fromarray(a[1]).convert('L')
#     b = Image.fromarray(a[2]).convert('L')
#     img = Image.merge("RGB", (r, g, b))
#     img = img.transpose(Image.ROTATE_270)
#     iconpath = './nyu_images/' + str(i) + '.jpg'
#     img.save(iconpath, optimize=True)

import numpy as np
import h5py
import os
from PIL import Image

f = h5py.File("/home/asus/Documents/4T/zyq/datasets/Depth/nyu_depth_v2_labeled.mat")
depths = f["depths"]
depths = np.array(depths)

# path_converted = './nyu_depths/'
# if not os.path.isdir(path_converted):
#     os.makedirs(path_converted)

max = depths.max()
print(depths.shape)
print(depths.max())
print(depths.min())

depths = depths / max * 255
depths = depths.transpose((0, 2, 1))

print(depths.max())
print(depths.min())

# for i in range(len(depths)):
#     print(str(i) + '.png')
#     depths_img = Image.fromarray(np.uint8(depths[i]))
#     depths_img = depths_img.transpose(Image.FLIP_LEFT_RIGHT)
#     iconpath = path_converted + str(i) + '.png'
#     depths_img.save(iconpath, 'PNG', optimize=True)

# # 同样方法可以提取rawdepth 对比查看深度图修复效果

# import scipy
# import numpy as np
# import h5py
# import os
# import torch
# from PIL import Image
# import cv2
# from torchvision import utils as vutils
#
# f = h5py.File("/home/asus/Documents/4T/zyq/Download/nyu_depth_v2_labeled.mat")
# labels = f["labels"]
# labels = np.array(labels)
# a = np.max(labels)
# path_converted = './nyu_labels_self_r/'
# if not os.path.isdir(path_converted):
#     os.makedirs(path_converted)
#
# labels_number = []
# for i in range(len(labels)):
#     labels_number.append(labels[i])
#     labels_0 = np.array(labels_number[i])
#     label_img = labels_number[i]
#     label_img = np.rot90(label_img, 3)
#     label_img = torch.from_numpy(label_img.astype(float))
#     # label_img = Image.fromarray(np.uint8(labels_number[i]))
#     # label_img = label_img.transpose(Image.ROTATE_270)
#
#     iconpath = './nyu_labels_self_r/' + str(i) + '.png'
#     # label_img.save(iconpath, 'PNG', optimize=True)
#     vutils.save_image(label_img, iconpath, normalize=True)

# import numpy as np
# import h5py
# import os
# from PIL import Image
#
# f = h5py.File("/home/asus/Documents/4T/zyq/Download/nyu_depth_v2_labeled.mat")
# labels = f["labels"]
# labels = np.array(labels)
#
# path_converted = './nyu_labels/'
# if not os.path.isdir(path_converted):
#     os.makedirs(path_converted)
#
# labels_number = []
# for i in range(len(labels)):
#     labels_number.append(labels[i])
#     labels_0 = np.array(labels_number[i])
#     label_img = Image.fromarray(np.uint16(labels_number[i]))
#     label_img = label_img.transpose(Image.ROTATE_270)
#
#     iconpath = './nyu_labels/' + str(i) + '.png'
#     label_img.save(iconpath, 'PNG', optimize=True)

# 语义分割图片 识别出了图片中的每个物体


# # 语义分割图片 识别出了图片中的每个物体

# import h5py
#
# f = h5py.File("./nyu_depth_v2_labeled.mat")
#
# ft = open('names.txt', 'w+')
# print(f["names"].shape)  # 打印查看类别个数，共894类
# for j in range(894):
#     name = f["names"][0][j]
#     obj = f[name]
#     strr = "".join(chr(i) for i in obj[:])
#     ft.write(strr + '\n')
#
# ft.close()

# # 从mat文件提取labels
# # 需要注意这个文件里面的格式和官方有所不同，长宽需要互换，也就是进行转置
# import cv2
# import scipy.io as scio
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# import os
#
# dataFile = '/home/asus/Documents/4T/zyq/Download/nyuv2-meta-data-master/labels40.mat'
# data = scio.loadmat(dataFile)
# labels = np.array(data["labels40"])
#
# path_converted = './nyu_labels40'
# if not os.path.isdir(path_converted):
#     os.makedirs(path_converted)
#
# labels_number = []
# for i in range(1449):
#     labels_number.append(labels[:, :, i].transpose((1, 0)))  # 转置
#     labels_0 = np.array(labels_number[i])
#     # print labels_0.shape
#     print(type(labels_0))
#     label_img = Image.fromarray(np.uint8(labels_number[i]))
#     # label_img = label_img.rotate(270)
#     label_img = label_img.transpose(Image.ROTATE_270)
#
#     iconpath = './nyu_labels40/' + str(i) + '.png'
#     label_img.save(iconpath, optimize=True)

# import scipy.io as scio
# dataFile = '/home/asus/Documents/4T/zyq/Download/nyuv2-meta-data-master/classMapping40.mat'
# data = scio.loadmat(dataFile)
# for i in range(40):
#     name = data['className'][0][i][0]
#     f1 = open("/home/asus/Desktop/name_40.txt", 'a', encoding='UTF-8')
#     f1.write('\'' + str(name).replace('\n', '') + '\',')


# spilt dataset
# import scipy.io as scio
# import os
# import shutil
# dataFile = '/home/asus/Documents/4T/zyq/Download/nyuv2-meta-data-master/splits.mat'
# data = scio.loadmat(dataFile)
#
# file_dir = '/home/asus/Documents/4T/zyq/datasets/Depth/nyu_v2_pic/nyu_images'
# new_file_dir = '/home/asus/Documents/4T/zyq/datasets/Depth/nyu_v2_pic/train/nyu_images'
# os.makedirs(new_file_dir)
# file_list = os.listdir(file_dir)
#
# file_dir_depth = '/home/asus/Documents/4T/zyq/datasets/Depth/nyu_v2_pic/nyu_depths'
# new_file_dir_depth = '/home/asus/Documents/4T/zyq/datasets/Depth/nyu_v2_pic/train/nyu_depths'
# os.makedirs(new_file_dir_depth)
# file_list_depth = os.listdir(file_dir_depth)
#
# file_dir_seg = '/home/asus/Documents/4T/zyq/datasets/Depth/nyu_v2_pic/nyu_labels40'
# new_file_dir_seg = '/home/asus/Documents/4T/zyq/datasets/Depth/nyu_v2_pic/train/nyu_labels40'
# os.makedirs(new_file_dir_seg)
# file_list_seg = os.listdir(file_dir_seg)
# for i in range(795):
#     name = data['trainNdxs'][i][0] - 1
#     for image in file_list:
#         if image == str(name) + ".jpg":
#             shutil.copy(os.path.join(file_dir, image), os.path.join(new_file_dir))
#
#     for depth in file_list_depth:
#         if depth == str(name) + ".png":
#             shutil.copy(os.path.join(file_dir_depth, depth), os.path.join(new_file_dir_depth))
#     for seg in file_list_seg:
#         if seg == str(name) + ".png":
#             shutil.copy(os.path.join(file_dir_seg, seg), os.path.join(new_file_dir_seg))
