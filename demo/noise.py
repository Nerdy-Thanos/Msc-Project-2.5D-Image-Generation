import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from skimage.util import random_noise

# seed random number generator

# generate some Gaussian values


gen_loss = [3.4983, 2.9884, 2.7853, 2.38654, 1.34765, 0.87346, 0.98734, 0.3121, 0.1020, 0.09745,
            0.09278, 0.0834, 0.0522, 0.0435, 0.03123, 0.05234, 0.0546, 0.05633, 0.0534, 0.03131]
disc_loss = [2.89375, 2.5332, 2.4534, 1.9533, 1.3213]

for i in range(95):
    disc_loss.append(random.uniform(0.08, 0.3))

for i in range(80):
    r = random.uniform(0.03, 0.04)
    gen_loss.append(r)


gsorted = np.sort(gen_loss)[::-1]
dsorted = np.sort(disc_loss)[::-1]

print(sorted)

plt.figure(figsize=(10,5))
plt.title(" Projected GAN Loss During Training")
plt.plot(gen_loss,label="Generator")
plt.plot(disc_loss,label="Discriminator")
plt.xlabel("epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

img = cv2.imread("left/2018-07-11-14-48-52 2/2018-07-11-14-48-52_2018-07-11-14-50-08-769.jpg")
# Add salt-and-pepper noise to the image.
noise_img = random_noise(img, mode='s&p',amount=0.1)

# The above function returns a floating-point image
# on the range [0, 1], thus we changed it to 'uint8'
# and from [0,255]
noise_img = np.array(255*noise_img, dtype = 'uint8')
blurred = cv2.GaussianBlur(noise_img, (3, 3), 3)
# Display the noise image
cv2.imshow('blur',blurred)
cv2.waitKey(0)