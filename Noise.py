def MES_PSNR(image1,image2):
    data1 = np.asarray(image1)
    data2 = np.asarray(image2)
    MSE = np.square(np.subtract(data1,data2)).mean()
    PSNR = 20 * log10(255/sqrt(MSE))
    print('mse of the embedded processing is :', MSE)
    print('psrn of the embedded processing is :', PSNR)
    return MSE,PSNR
def S_SIM(image_Ogirinal,image_Embedded):
   grayA = cv2.cvtColor(image_Ogirinal, cv2.COLOR_BGR2GRAY)
   grayB = cv2.cvtColor(image_Embedded, cv2.COLOR_BGR2GRAY)
   (score, diff) = ssim.compare_ssim(grayA, grayB, full=True)
   diff = (diff * 255).astype("uint8")
   print("SSIM of the embedded processing is : {}".format(score))
   return score

def randomnoise(image):
    # Add salt-and-pepper noise to the image.
    noise_img = random_noise(image, mode='s&p', amount=0.3)
    # The above function returns a floating-point image
    # on the range [0, 1], thus we changed it to 'uint8'
    # and from [0,255]
    noise_img = np.array(255 * noise_img, dtype='uint8')

    # Display the noise image
    cv2.imshow('noise_image.png', noise_img)
    cv2.imwrite("noise_image.png",noise_img)
    cv2.waitKey(0)
    return noise_img
