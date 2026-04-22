import numpy as np 
import matplotlib.pyplot as plt 

image = plt.imread("gambar_contoh.jpeg")

R = image[:, :, 0]
G = image[:, :, 1]
B = image[:, :, 2]

#method luminosity
gray = 0.299 * R + 0.587 * G + 0.114 * B
#nilai gray totalnya akan berada diantara 0-255, karena nilai R,G,B berada diantara 0-255

plt.imshow(gray, cmap='gray')
plt.axis('off')
plt.show()

# nilai kecil → ditampilkan hitam

# nilai sedang → abu-abu

# nilai besar → putih

