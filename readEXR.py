import numpy as np
import OpenEXR as exr
import Imath
import matplotlib.pyplot as plt

exrfile = exr.InputFile('/home/asus/Documents/4T/zyq/Download/datasets/synscapes_test/depth/annotation/4.exr')
header = exrfile.header()
dw = header['dataWindow']
issize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
channelData = dict()
for c in header['channels']:
    C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
    C = np.frombuffer(C, dtype=np.float32)
    C = np.reshape(C, issize)

    channelData[c] = C
# colorChannels = ['R', 'G', 'B', 'A'] if 'A' in header['channels'] else ['R', 'G', 'B']
# img = np.concatenate([channelData[c][..., np.newaxis] for c in colorChannels], axis=2)
# img[..., :3] = np.where(img[..., :3] <= 0.0031308,
#                         12.92 * img[..., :3],
#                         1.055 * np.power(img[..., :3], 1 / 2.4) - 0.055)
# img = np.where(img < 0.0, 0.0, np.where(img > 1.0, 1, img))
Z = channelData['Z']
# Z = np.clip(Z * 80, Z.min(), Z.max())
print(Z.max())
print(Z.min())

plt.imshow(Z, cmap='gray')
plt.axis('off')
plt.show()
