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

path1 = 'D:\Imageabc'
data_path1 = os.path.join(path1,'*g')
files = glob.glob(data_path1)
data1 = []
p1 = 0
for f2 in files:
    p1 = p1 + 1
    image1 = cv2.imread(f2)
    anh = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    x = asarray(anh)
    demension1 = x.shape
    c = demension1[0]
    d = demension1[1]
    m = np.zeros([c, d])
    for i in range(c):
        for j in range(d):
            if x[i, j] % 2 == 0:
                m[i, j] = 0
            else:
                m[i, j] = 1
    print('thông tin trích là:',m)
