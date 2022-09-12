import numpy as np
import matplotlib.pyplot as plt
import cv2
import pywt
from PIL import Image
import pywt.data
import os
import glob


# Cover Image : C
# Watermark Image : W
# Watermarked Image : M

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
    return resized,dim
# Load image
arr = os.listdir('C:/Users/USER/Desktop/DWT/Img')
path = "C:/Users/USER/Desktop/DWT/DWT_Emb"
img_dir = "C:/Users/USER/Desktop/DWT/Img"
data_path = os.path.join(img_dir, '*g')
files = glob.glob(data_path)
data = []

i = 0
for f1 in files:
    i = i + 1
    original1 = cv2.imread(f1)
    data.append(f1)
    #original = cv2.resize(original, dim, interpolation=cv2.INTER_AREA)
    original,dim = resizeImage(original1)

    watermark = cv2.imread('little.png')
    watermark = rgb2gray(watermark)
    watermark = cv2.resize(watermark, dim, interpolation=cv2.INTER_AREA)
    # 3 - DWT Watermark Image
    Wcoeffs = pywt.dwt2(watermark, 'haar')
    WLL, (WLH, WHL, WHH) = Wcoeffs

    Wcoeffs1 = pywt.dwt2(WLL, 'haar')
    WLL1, (WLH1, WHL1, WHH1) = Wcoeffs1

    Wcoeffs2 = pywt.dwt2(WLL1, 'haar')
    WLL2, (WLH2, WHL2, WHH2) = Wcoeffs2

    # 3 - DWT Cover Image
    Ccoeffs = pywt.dwt2(original, 'haar')
    CLL, (CLH, CHL, CHH) = Ccoeffs

    Ccoeffs1 = pywt.dwt2(CLL, 'haar')
    CLL1, (CLH1, CHL1, CHH1) = Ccoeffs1

    Ccoeffs2 = pywt.dwt2(CLL1, 'haar')
    CLL2, (CLH2, CHL2, CHH2) = Ccoeffs2

    # @ technical
    k = 0.99
    q = 0.009
    x = 0.99
    WMI = k * (CLL2) + q * (WLL2)
    #print(WMI.shape)
    '''
    # 2D multilevel reconstruction using waverec 2
    coeffs_re = (WMI,(CLH2,CHL2,CHH2),(CLH1,CHL1,CHH1),(CLH,CHL,CHH))
    coeffs_re = pywt.waverec2(coeffs_re,'haar', mode='symmetric')

    '''
    a1 = x * CLH2 + (1 - x) * WLH2
    a2 = x * CHL2 + (1 - x) * WHL2
    a3 = x * CHH2 + (1 - x) * WHH2

    a = (WMI, (a1, a2, a3))
    # a = (WMI,(CLH,CHL,CHH))
    aa = pywt.idwt2(a, 'haar')
    #print(aa.shape)

    b1 = x * CLH1 + (1 - x) * WLH1
    b2 = x * CHL1 + (1 - x) * WHL1
    b3 = x * CHH1 + (1 - x) * WHH1

    b = (aa, (b1, b2, b3))
    bb = pywt.idwt2(b, 'haar')
    #print(bb.shape)

    c1 = x * CLH + (1 - x) * WLH
    c2 = x * CHL + (1 - x) * WHL
    c3 = x * CHH + (1 - x) * WHH

    c = (bb, (c1, c2, c3))
    cc = pywt.idwt2(c, 'haar')
    #print(cc.shape)
    cc = Image.fromarray(cc)
    #cc.show()
    if cc != 'RGB':
        cc = cc.convert('RGB')
    cc.save(os.path.join(path, 'DWT_{0}'.format(arr[i-1])))
