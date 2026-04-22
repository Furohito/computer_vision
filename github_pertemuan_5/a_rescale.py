#READING, DOWNSIZING AND SAVING BACK A COLOR IMAGE
#Attribution: Mohammad Nasucha, Ph.D.
#A file named "Gambar.jpg" will be downsized and saved as 
# "Gambar_r.jpg" in the same folder.
#If the image is already small, the downsizing code won't be executed
#and an error message will show up.

import numpy as np
import matplotlib.pyplot as plt
#import warnings
#warnings.filterwarnings("ignore")

#USER ENTRIES
row_res = 500                              #Ukuran gambar yang ditargetkan setelah resizing.
dir = "C:/Users/rende/OneDrive/Dokumen/data kuliah/semester 6/6 computer vision/pertemuan 5/tugas/"                                       #Direktori tempat gambar input disimpan.
file = "labubu"

#FUNCTION TO RESIZE
def resize (dir, file, row_res):
    path = dir + file + ".jpeg"
    pic = plt.imread(path)
    pic = pic.astype(int)
    row, col, ch = pic.shape
    #If the image size <= row_res, do not resize it.
    if (row < row_res+1):
        print("The image is small enough thus it's not resized.")
        pic_res = pic

    # If the image size > row_res, resize it.
    if row > row_res:
        res_factor = row / row_res
        col_res = int(round(col/res_factor))
        # Creating a template array for the output image:
        pic_res = np.zeros(shape=(row_res, col_res, 3), dtype=np.uint8)
        for m_res in range(0, row_res-1):
            print(m_res)
            for n_res in range(0, col_res-1):
                m = int(round(res_factor * m_res))  # Mapping pixel coordinates betw pic_res and pic
                n = int(round(res_factor * n_res))  # Mapping pixel coordinates betw pic_res and pic
                pic_res[m_res, n_res, :] = pic[m, n, :]  # Copying pixels values of pic to pic_res, according to the mapping

    path = dir + file + "_r" + ".jpg"
    plt.imsave(path, pic_res)

#MAIN PROGRAM
resize (dir, file, row_res)

#from skimage.io import imread, imsave