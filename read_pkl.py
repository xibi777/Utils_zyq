import pickle
fr = open("/home/asus/Documents/4T/zyq/PycharmProjects/Segformer_test/SegFormer-master/tools/work_dirs/res.pkl", 'rb')  # open的参数是pkl文件的路径
inf = pickle.load(fr)  # 读取pkl文件的内容
print(inf)
fr.close()
