print("\033c")

import numpy as np
import matplotlib.pyplot as plt

rowmax = int(1079)
colmax = int(1919)
batas1 = int(0.2*colmax)
batas2 = int(0.4*colmax)
batas3 = int(0.6*colmax)
batas4 = int(0.8*colmax)

Gambar = np.zeros(shape=(rowmax+1, colmax+1, 3), dtype=np.uint8)
for i in range(0, rowmax+1):
    for j in range(0, colmax+1):
        if (j>=0 and j<batas1):
            Gambar[i, j, 0] = 255
            Gambar[i, j, 2] = 255
        elif (j>=batas1 and j<batas2):
            Gambar[i, j, 0] = 255
            Gambar[i, j, 1] = 255
        elif (j>=batas2 and j<batas3):
            Gambar[i, j, 2] = 255
        elif (j>=batas3 and j<batas4):
            Gambar[i, j, 1] = 255
        else:
            Gambar[i, j, 0] = 255

print(type(Gambar))
plt.figure()
plt.imshow(Gambar)
plt.show()

# plt.imsave("pelangi_vertikal.png", Gambar)
# plt.imsave("pelangi_vertikal.jpg", Gambar)