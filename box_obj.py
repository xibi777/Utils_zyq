import numpy as np

box = np.load(
    '/media/xibi/caca8aa9-2ed6-4184-b103-26aa20773f10/zyq/3D_work/3D-SPS_data/scannet_train_images/out_3D_box_select/scene0000_00/280/4_aligned_bbox.npy')
center = box[:3]
dimensions = box[3:6]
# 首先，我们需要计算出箱子的8个顶点
# center = np.array([cx, cy, cz])  # 箱子中心
# dimensions = np.array([width, height, depth])  # 箱子的长宽高

half_dims = dimensions / 2

# vertices 顶点
vertices = np.array([
    [-1, -1, -1],
    [-1, -1, 1],
    [-1, 1, -1],
    [-1, 1, 1],
    [1, -1, -1],
    [1, -1, 1],
    [1, 1, -1],
    [1, 1, 1]
]) * half_dims

# 将顶点相对于中心进行偏移
vertices += center

# faces 面
faces = [
    [1, 3, 7, 5],
    [2, 6, 8, 4],
    [1, 2, 4, 3],
    [5, 7, 8, 6],
    [1, 5, 6, 2],
    [3, 4, 8, 7]
]

# 打开要写入的文件
box_name = 'box_280_4.obj'
box_path = '/home/xibi/Desktop/3D_pesudo_label_visualization/box'
box_file = box_path + '/' + box_name
with open(box_file, 'w') as f:
    # 写入顶点
    for vertex in vertices:
        f.write('v {} {} {}\n'.format(*vertex))
    # 写入面
    for face in faces:
        f.write('f {} {} {} {}\n'.format(*face))
