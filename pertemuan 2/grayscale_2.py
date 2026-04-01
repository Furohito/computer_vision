import numpy as np
from matplotlib import pyplot as plt

img = plt.imread("image.png")

R = img[:, :, 0]
G = img[:, :, 1]
B = img[:, :, 2]

# Luminosity method
gray = 0.299*R + 0.587*G + 0.114*B

gray2 = (R + G + B)/3

# Normalization
gray = gray/255
gray2 = gray2/255

plt.figure(1)
plt.imshow(gray, cmap='gray')
plt.title("Luminosity Method")

plt.figure(2)
plt.imshow(gray2, cmap='gray')
plt.title("Average Method")
plt.show()