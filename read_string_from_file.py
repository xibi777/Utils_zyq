# 一行一行读取txt中的数字并进行累加

f = open("/home/asus/Desktop/loss_dice_0059999.txt")
line = f.readline()
a = 0
s = 0
while line:
    s = s + float(line)
    line = f.readline()
    a = a + 1
f.close()
print(a)
print(s)
