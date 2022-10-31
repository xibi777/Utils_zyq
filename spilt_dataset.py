import os
import random
import shutil

random.seed(1)
res = random.sample(range(1, 25001), 20000)
file_dir = '/home/asus/Documents/4T/zyq/datasets/Synscapes/img/rgb/'
new_file_dir = '/home/asus/Documents/4T/zyq/Download/datasets/synscapes/depth/train'
os.makedirs(new_file_dir)
file_list = os.listdir(file_dir)

file_dir_depth = '/home/asus/Documents/4T/zyq/datasets/Synscapes/img/depth/'
new_file_dir_depth = '/home/asus/Documents/4T/zyq/Download/datasets/synscapes/depth/annotation'
os.makedirs(new_file_dir_depth)
file_list_depth = os.listdir(file_dir_depth)
for i in range(len(res)):
    for image in file_list:
        if image == str(res[i]) + ".png":
            shutil.copy(os.path.join(file_dir, image), os.path.join(new_file_dir))
    for depth in file_list_depth:
        if depth == str(res[i]) + ".exr":
            shutil.copy(os.path.join(file_dir_depth, depth), os.path.join(new_file_dir_depth))

