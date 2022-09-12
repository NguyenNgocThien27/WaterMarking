import numpy as np
import matplotlib.pyplot as plt
import cv2
import pywt
from PIL import Image
import pywt.data
import os
import glob
from skimage.util import random_noise

arr = os.listdir('D:/Image')
path = "D:/Image_Extract"
img_dir1 = "D:/Image"
data_path1 = os.path.join(img_dir1, '*g')
files1 = glob.glob(data_path1)
data1 = []

img_dir2 = "D:/Imageabc"
data_path2 = os.path.join(img_dir2, '*g')
files2 = glob.glob(data_path2)
data2 = []
def randomnoise(image):
    # Add salt-and-pepper noise to the image.
    noise_img = random_noise(image, mode='s&p', amount=0.05)
    # The above function returns a floating-point image
    # on the range [0, 1], thus we changed it to 'uint8'
    # and from [0,255]
    noise_img = np.array(255 * noise_img, dtype='uint8')

    # Display the noise image
    cv2.imshow('noise_image.png', noise_img)
    cv2.imwrite("noise_image.png",noise_img)
    cv2.waitKey(0)
    return noise_img
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
def resizeImage(image):
    a = rgb2gray(image)
    b = a.shape
    print(b)
    b0 = b[0] % 8
    b1 = b[1] % 8
    if b0 >= 4:
        b2 = b[0] + (8 - b0)
    else:
        b2 = b[0] - b0

    if b1 >= 4:
        b3 = b[1] + (8 - b1)
    else:
        b3 = b[1] - b1

    dim = (b3, b2)
    print(dim)
    resized = cv2.resize(a, dim, interpolation=cv2.INTER_AREA)
    return resized
p = 0
# Alpha Technical
k = 0.99
q = 0.009
x = 0.99
for k in range(1000):
    p = p + 1
    original1 = files1[k]
    original = cv2.imread(original1)
    Watermarked = files2[k]
    Watermarked = cv2.imread(Watermarked)
    Watermarked = randomnoise(Watermarked)

    # Cover Image 3-DWT
    original = resizeImage(original)
    Ccoeffs = pywt.dwt2(original, 'haar')
    CLL, (CLH, CHL, CHH) = Ccoeffs

    Ccoeffs1 = pywt.dwt2(CLL, 'haar')
    CLL1, (CLH1, CHL1, CHH1) = Ccoeffs1

    Ccoeffs2 = pywt.dwt2(CLL1, 'haar')
    CLL2, (CLH2, CHL2, CHH2) = Ccoeffs2

    # Watermarked Image 3-DWT
    Watermarked = rgb2gray(Watermarked)

    Mcoeffs = pywt.dwt2(Watermarked, 'haar')
    MLL, (MLH, MHL, MHH) = Mcoeffs

    Mcoeffs1 = pywt.dwt2(MLL, 'haar')
    MLL1, (MLH1, MHL1, MHH1) = Mcoeffs1

    Mcoeffs2 = pywt.dwt2(MLL1, 'haar')
    MLL2, (MLH2, MHL2, MHH2) = Mcoeffs2

    RW = (MLL2 - k * CLL2) / q
    d1 = (MLH2 - x * CLH2) / (1 - x)
    d2 = (MHL2 - x * CHL2) / (1 - x)
    d3 = (MHH2 - x * CHH2) / (1 - x)
    dd = pywt.idwt2((RW, (d1, d2, d3)), 'haar')

    e1 = (MLH1 - x * CLH1) / (1 - x)
    e2 = (MHL1 - x * CHL1) / (1 - x)
    e3 = (MHH1 - x * CHH1) / (1 - x)
    ee = pywt.idwt2((dd, (e1, e2, e3)), 'haar')

    f1 = (MLH - x * CLH) / (1 - x)
    f2 = (MHL - x * CHL) / (1 - x)
    f3 = (MHH - x * CHH) / (1 - x)
    ff = pywt.idwt2((ee, (f1, f2, f3)), 'haar')

    ff = Image.fromarray(ff)
    if ff != 'RGB':
        ff = ff.convert('RGB')
    ff.save(os.path.join(path, 'DWT_{0}'.format(arr[p-1])))
