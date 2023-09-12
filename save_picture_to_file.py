import os
from torchvision import utils as vutils
from
# solution 1
file_path = '/home/asus/Desktop/M2F_proposal/proposal/' + str(0) + '/'
os.makedirs(file_path)
save_pro = file_path + str(0) + '.png'
vutils.save_image(picture, save_pro, normalize=True)

# solution 2
# 保存后会有白边填充
plt.imshow(picture)
plt.axis('off')
plt.savefig(os.path.join(file_path, str(0)), pad_inches=0)
plt.show()
