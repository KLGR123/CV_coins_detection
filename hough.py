import numpy as np
import cv2
import tqdm

class HoughCricle:
    def __init__(self, img, origin_img, rad=100, filter_size=50, min_rad=45, max_rad=60):
        self.img = img # 经过 Canny 预处理
  
        self.rad = rad
        self.h = img.shape[0]
        self.w = img.shape[1]
        self.size = filter_size
        self.max_rad = max_rad
        self.min_rad = min_rad

        self.array = np.zeros((self.h, self.w, rad))
        self.filter3D = np.ones((self.size, self.size, rad))

        self.output = origin_img

    def fill_array(self, x0, y0, radius):
        x = radius
        y = 0
        decision = 1 - x
        
        while y < x:
            if x + x0 < self.h and y + y0 < self.w:
                self.array[x + x0, y + y0, radius] += 1
            if y + x0 < self.h and x + y0 < self.w:
                self.array[y + x0, x + y0, radius] += 1
            if -x + x0 < self.h and y + y0 < self.w:
                self.array[-x + x0, y + y0, radius] += 1
            if -y + x0 < self.h and x + y0 < self.w:
                self.array[-y + x0, x + y0, radius] += 1
            if -x + x0 < self.h and -y + y0 < self.w:
                self.array[-x + x0, -y + y0, radius] += 1
            if -y + x0 < self.h and -x + y0 < self.w:
                self.array[-y + x0, -x + y0, radius] += 1
            if x + x0 < self.h and -y + y0 < self.w:
                self.array[x + x0, -y + y0, radius] += 1
            if y + x0 < self.h and -x + y0 < self.w:
                self.array[y + x0, -x + y0, radius] += 1

            y += 1
            if decision <= 0:
                decision += 2 * y + 1
            else:
                x -= 1
                decision += 2 * (y - x) + 1

    def detect(self):
        edges = np.where(self.img == 255)
        for i in tqdm.tqdm(range(len(edges[0]))):
            x = edges[0][i]
            y = edges[1][i]
            for radius in range(self.min_rad, self.max_rad):
                self.fill_array(x, y, radius)
                    
        plist = rlist = []

        for i in tqdm.tqdm(range(0, self.h - self.size, self.size)):
            for j in range(0, self.w - self.size, self.size):
                self.filter3D = self.array[i:i+self.size, j:j+self.size, :] * self.filter3D
                point = np.where(self.filter3D == self.filter3D.max())
                try:
                    if(self.filter3D.max() > 90):
                        cv2.circle(self.output, (int(point[1] + j), int(point[0] + i)), int(point[2]), (0, 255 ,0), 2)
                        plist.append((int(point[1] + j), int(point[0] + i)))
                        rlist.append(int(point[2]))

                    self.filter3D[:, :, :] = 1    
                except:
                    continue

        cv2.imshow('Detected circle', self.output)
        return self.output, plist, rlist