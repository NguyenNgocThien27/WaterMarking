from xlwt import Workbook
from PIL import Image
from numpy import asarray
from math import log10,sqrt
from skimage.util import random_noise
from skimage.measure import _structural_similarity as ssim

import cv2
import numpy as np
import matplotlib.image as img
import os
import glob
import xlwt

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

arr = os.listdir('C:/Users/USER/Desktop/LSB/Img1')
path = 'D:\Imageabc'
img_dir = "C:/Users/USER/Desktop/LSB/Img1" # Enter Directory of all images
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
p = 0
for f1 in files:
    p = p + 1
    image = cv2.imread(f1)
    image = rgb2gray(image)
    data = asarray(image)
    a = (np.ceil(data))
    demension = image.shape
    print(demension)
    b = np.random.randint(0, 2, demension)
    n = np.zeros(demension)
    for i in range(demension[0]):
        for j in range(demension[1]):
            if b[i,j] == 0:
                if a[i, j] % 2 == 0:
                    n[i, j] = a[i, j]
                else:
                    n[i, j] = a[i, j] - 1
            else:
                 if a[i, j] % 2 == 0:
                     n[i, j] = a[i, j] + 1
                 else:
                     n[i, j] = a[i, j]
    print('Thông tin nhúng là:',b)
img12 = Image.fromarray(n)


if img12 != 'RGB':
    img12 = img12.convert('RGB')
img12.save(os.path.join(path, 'LSB_{0}'.format(arr[p-1])))
