import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import os

# 将 rgb 图像转化为灰度图像
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

# 从指定路径读入 img 图片
def load_data(img, gray=False): 
    img = mpimg.imread(img)
    if gray == True:
        img = rgb2gray(img) 
    return img

# 图片可视化
def visualize(img, format=None):
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    
    plt.figure(figsize=(20, 20))
    plt.imshow(img, format)
    plt.show()

# 判断邻居像素是否强像素
def neighbor_strong(Z, M, N, weak, strong):
    for i in range(1, M-1):
        for j in range(1, N-1):
            if Z[i,j] == weak:
                if (Z[i+1, j-1] == strong) or (Z[i+1, j] == strong) or (Z[i+1, j+1] == strong) \
                or (Z[i, j-1] == strong) or (Z[i, j+1] == strong) or (Z[i-1, j-1] == strong) \
                or (Z[i-1, j] == strong) or (Z[i-1, j+1] == strong):
                    Z[i, j] = strong
                else:
                    Z[i, j] = 0

    return Z