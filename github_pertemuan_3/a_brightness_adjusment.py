import numpy as np
import matplotlib.pyplot as plt

#User entries
nama_file = 'bunga'
delta_Intens = 50 #intensitas ubah ke 50 dan -50

#Main program
gambar_asal = plt.imread(nama_file+'.jpeg')
row, col, depth = gambar_asal.shape
plt.figure('Gambar Asal')
plt.imshow(gambar_asal)

 
#hasil brightnes
gambar = gambar_asal.copy()
print(gambar[100,100])

for y in range (0, row):
    for x in range (0, col):
        r = float(gambar[y, x, 0]) + delta_Intens
        r = max(r, 0)
        g = float(gambar[y, x, 1]) + delta_Intens
        g = max(g, 0)
        b = float(gambar[y, x, 2]) + delta_Intens
        b = max(b, 0)
        gambar[y, x, 0] = r
        gambar[y, x, 1] = g
        gambar[y, x, 2] = b 

# gambar [:, :, :] = gambar [:, :, :] + delta_Intens
plt.figure('Gambar Hasil')
plt.imshow(gambar)
plt.show()

#check tipe data
# print(gambar[100,100])
# # print('Tipe data gambar asal : ', gambar_asal.dtype)
# # print('Tipe data gambar hasil : ', gambar.dtype)
# #int gabisa negatif

