print("\033c")

import numpy as np
import matplotlib.pyplot as plt

rowmax = int(1079)
colmax = int(1919)
batas1 = int(0.2*rowmax)
batas2 = int(0.4*rowmax)
batas3 = int(0.6*rowmax)
batas4 = int(0.8*rowmax)

Gambar = np.zeros(shape=(rowmax+1, colmax+1, 3), dtype=np.uint8)
for i in range(0, rowmax+1):
    for j in range(0, colmax+1):
        if (i>=0 and i<batas1):
            Gambar[i, j, 0] = 255 #warna merah
        elif (i>=batas1 and i<batas2):
            Gambar[i, j, 1] = 255 #warna hijau
        elif (i>=batas2 and i<batas3):
            Gambar[i, j, 2] = 255 #warna biru
        elif (i>=batas3 and i<batas4): #warna kuning
            Gambar[i, j, 0] = 255 
            Gambar[i, j, 1] = 255 
        else:                           #warna ungu
            Gambar[i, j, 0] = 255 
            Gambar[i, j, 2] = 255

print(type(Gambar))
plt.figure()
plt.imshow(Gambar)
plt.show()

# plt.imsave("pelangi_horizontal.png", Gambar)
# plt.imsave("pelangi_horizontal.jpg", Gambar)