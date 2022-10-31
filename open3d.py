import open3d as o3d
pcd = o3d.io.read_point_cloud('/home/asus/Desktop/depth2Cloud/coco/point_clouds/000000000785.ply')
o3d.visualization.draw_geometries([pcd])
