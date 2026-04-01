print("\033c")
import matplotlib.pyplot as plt
import numpy as np 

row = round(1080)
col = round(1920)

row1 = round(0.25*row)
row2 = round(0.5*row)
row3 = round(0.75*row)
col1 = round(0.25*col)
col2 = round(0.75*col)

Gambar = np.zeros(shape=(row, col, 3), dtype=np.uint8)

Gambar [row1:row2+1, col1:col2+1, 0] = 255
Gambar [row2:row3+1, col1:col2+1, 0:3] = 255

plt.figure()
plt.imshow(Gambar)
plt.show()

plt.imsave("bendera_merah_putih.png", Gambar)
plt.imsave("bendera_merah_putih.jpg", Gambar)
