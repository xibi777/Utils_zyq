import os
import shutil
from numpy import random
import mmcv


def hue(img):
    """Hue distortion."""
    if random.randint(2):
        img = mmcv.bgr2hsv(img)
        img[:, :, 0] = (img[:, :, 0].astype(int) + random.randint(-18, 18)) % 180
        img = mmcv.hsv2bgr(img)
    return img


if __name__ == '__main__':
    file_dir = '/home/asus/Documents/4T/zyq/datasets/Depth/nyu/training'
    new_file_dir = '/home/asus/Documents/4T/zyq/datasets/Depth/nyu_diy/depth'
    file_client_args = {'backend': 'disk'}
    file_client = mmcv.FileClient(**file_client_args)

    file_list = os.listdir(new_file_dir)
    for file in file_list:
        if file.find('furniture_store') != -1:
            os.remove(os.path.join(new_file_dir, file))

