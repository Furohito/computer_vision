import numpy as np
import matplotlib.pyplot as plt

#User entries
nama_file = 'bunga'
delta_Contrast = float(30) #intensitas ubah ke 50 dan -50
threshold = 127 #threshold ubah ke 127 dan 100

#Main program
gambar_asal = plt.imread(nama_file+'.jpeg')
row, col, depth = gambar_asal.shape
plt.figure('Gambar Asal')
plt.imshow(gambar_asal)

 
#hasil contrast
gambar = gambar_asal.copy()
print(gambar[100,100])

for y in range (0, row):
    for x in range (0, col):
        if(float(gambar[y, x, 0]) + float(gambar[y, x, 1]) + float(gambar[y, x, 2])) /3 > threshold:
            gambar[y, x, 0] = min (float((gambar[y, x, 0]) + delta_Contrast), 255)
            gambar[y, x, 1] = min (float((gambar[y, x, 1]) + delta_Contrast), 255)
            gambar[y, x, 2] = min (float((gambar[y, x, 2]) + delta_Contrast), 255)
        if (float(gambar[y, x, 0]) + float(gambar[y, x, 1]) + float(gambar[y, x, 2])) /3 <= threshold:
            gambar[y, x, 0] = max (float((gambar[y, x, 0]) - delta_Contrast), 0)
            gambar[y, x, 1] = max (float((gambar[y, x, 1]) - delta_Contrast), 0)
            gambar[y, x, 2] = max (float((gambar[y, x, 2]) - delta_Contrast), 0)

# gambar [:, :, :] = gambar [:, :, :] + delta_Intens
plt.figure('Gambar Hasil')
plt.imshow(gambar)
plt.show()

#check tipe data
# print(gambar[100,100])
# # print('Tipe data gambar asal : ', gambar_asal.dtype)
# # print('Tipe data gambar hasil : ', gambar.dtype)
# #int gabisa negatif

