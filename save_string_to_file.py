# 将字符串、数字存入txt中（换行）

a = 1
loss_mask = float(a)
f1 = open("/home/asus/Desktop/M2F_result/loss_mask_274999.txt", 'a', encoding='UTF-8')
f1.write(str(loss_mask) + '\n')
f1.close()
