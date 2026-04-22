import numpy as np
import matplotlib.pyplot as plt

#User Entries
nama_file = 'Gambar_contoh.jpeg'

#Main Code
pic = plt.imread(nama_file)
plt.figure(1)
plt.imshow(pic) 
plt.axis('off')

row, col, ch = pic.shape

#GRAYSCALE
hasil1 = np.zeros((row, col), dtype=np.uint8)
for y in range(0, row):
    print(y)
    for x in range(0, col):
        average = (float(pic[y, x, 0]) + float(pic[y, x, 1]) + float(pic[y, x, 2])) / 3
        hasil1[y, x] = average

plt.figure(2)
plt.imshow(hasil1, cmap='gray')
plt.axis('off')
plt.show()

#NORMALIZATION
hasil2 = np.zeros((row, col), dtype=np.uint8)
for y in range(0, row):
    print(y)
    for x in range(0, col):
        hasil2[y, x] =  hasil1[y, x] / 255 

plt.figure(3)
plt.imshow(hasil2, cmap='gray')
plt.axis('off')

print(hasil1[100,100])
print(hasil2[100,100])
plt.show()