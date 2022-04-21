import numpy as np
from scipy import ndimage
from scipy.ndimage.filters import convolve 
from utils import neighbor_strong

class CannyEdge:
    def __init__(self, img, sigma=1, kernel_size=5, weak_pixel=75, strong_pixel=255, lowthreshold=0.05, highthreshold=0.15):
        self.img = img
  
        self.weak_pixel = weak_pixel
        self.strong_pixel = strong_pixel

        self.sigma = sigma
        self.kernel_size = kernel_size

        self.lowThreshold = lowthreshold
        self.highThreshold = highthreshold
    
    # 二维高斯核
    def gaussian_kernel(self, size, sigma=1):
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        norm = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * norm
        return g
    
    # Sobel 算子
    def sobel_filters(self, img):
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

        Ix = ndimage.filters.convolve(img, Kx)
        Iy = ndimage.filters.convolve(img, Ky)

        G = np.hypot(Ix, Iy)
        G = G / G.max() * 255
        
        theta = np.arctan2(Iy, Ix)
        return (G, theta)
    
    # 非最值抑制，实现降噪
    def non_max_suppression(self, img, D):
        M, N = img.shape
        Z = np.zeros((M, N), dtype=np.int32)
        
        angle = D * 180. / np.pi
        angle[angle < 0] += 180

        for i in range(1, M - 1):
            for j in range(1, N - 1):
                q = 255
                r = 255
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                elif 22.5 <= angle[i,j] < 67.5:
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                elif 67.5 <= angle[i,j] < 112.5:
                    q = img[i+1, j]
                    r = img[i-1, j]
                elif 112.5 <= angle[i,j] < 157.5:
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0
        return Z

    # 滞后门限
    def threshold_hysteresis(self, img):
        highThreshold = img.max() * self.highThreshold
        lowThreshold = highThreshold * self.lowThreshold

        M, N = img.shape
        Z = np.zeros((M, N), dtype=np.int32)

        weak = np.int32(self.weak_pixel)
        strong = np.int32(self.strong_pixel)

        strong_i, strong_j = np.where(img >= highThreshold)
        zeros_i, zeros_j = np.where(img < lowThreshold)
        weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

        Z[strong_i, strong_j] = strong
        Z[weak_i, weak_j] = weak
        Z = neighbor_strong(Z, M, N, weak, strong)           
        return Z
    
    def detect(self): 
        img_smoothed = convolve(self.img, self.gaussian_kernel(self.kernel_size, self.sigma))
        gradient, theta = self.sobel_filters(img_smoothed)
        Img = self.non_max_suppression(gradient, theta)
        Img = self.threshold_hysteresis(Img)
        return Img
