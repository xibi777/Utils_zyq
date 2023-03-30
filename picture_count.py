import os

file_dir = "/home/asus/Documents/4T/zyq/datasets/Depth/nyu/training"
file_list = os.listdir(file_dir)
count = 0
for file in file_list:
    if os.path.isdir(file_dir + '/' + file):
        file_name_list = os.listdir(file_dir + '/' + file)
        for file_name in file_name_list:
            if file_name.count('.jpg'):
                count += 1

print(count)
