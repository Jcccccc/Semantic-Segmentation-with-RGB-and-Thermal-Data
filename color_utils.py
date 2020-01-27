import numpy as np
import cv2

def voc_colormap(N=256):
    bitget = lambda val, idx: ((val & (1 << idx)) != 0)
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= (bitget(c, 0) << 7 - j)
            g |= (bitget(c, 1) << 7 - j)
            b |= (bitget(c, 2) << 7 - j)
            c >>= 3

        cmap[i, :] = [r, g, b]
    return cmap

colormap = voc_colormap()

def color_predicts(img):
    # img = cv2.imread(label_path,cv2.CAP_MODE_GRAY)
    color = np.ones([img.shape[0], img.shape[1], 3], dtype=np.uint8)
    color[img==0] = np.array([0, 0, 0])
    color[img==1] = np.array([0, 0, 128])
    color[img==2] = np.array([0, 128, 0])
    color[img==3] = np.array([0, 128, 128])
    color[img==4] = np.array([128, 0, 0])
    color[img==5] = np.array([128, 0, 128])
    return np.array(color, dtype=np.uint8)

def color_annotation(label_path, output_path):

    img = np.array(Image.open(label_path))

    color = np.ones([img.shape[0], img.shape[1], 3])

    color[img==0] = [0, 0, 0]
    color[img==1] = colormap[1]
    color[img==2] = colormap[2]
    color[img==3] = colormap[3]
    color[img==4] = colormap[4]
    color[img==5] = colormap[5]

    cv2.imwrite(output_path,color)