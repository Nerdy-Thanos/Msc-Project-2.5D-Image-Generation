import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
imgL = cv.imread('/Users/vignesh/Documents/GitHub/Msc-Project-2.5D-Image-Generation/gan_left.png',0)
imgR = cv.imread('/Users/vignesh/Documents/GitHub/Msc-Project-2.5D-Image-Generation/gan_right.png',0)
stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()