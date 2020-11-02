import os
import glob
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pydicom

def loaddata():
    flag = 1
    imagePath="../LDCT-and-Projection-data"
    if not os.path.exists(imagePath):
        print(imagePath+"not exist")
        return
    allFulldose=glob.glob(imagePath+"/*/*/*-Full dose images-*/*.dcm")
    allLowdose=glob.glob(imagePath + "/*/*/*-Low dose images-*/*.dcm")
    def path2img(imgPaths):
        images=[]
        for imgPath in imgPaths:
            image_bytes = pydicom.dcmread(imgPath)

            image=image_bytes.pixel_array
            images.append(image)
        images=np.array(images)
        return images
    
    fulldose_imgs=path2img(allFulldose)
    lowdose_imgs=path2img(allLowdose)

    full_ds=tf.data.Dataset.from_tensor_slices(fulldose_imgs)
    low_ds=tf.data.Dataset.from_tensor_slices(lowdose_imgs)


    return low_ds,full_ds

l_ds,f_ds=loaddata()
for i,l in enumerate(l_ds):
    if i%2000==0:
        plt.imshow(l,cmap='gray')
        plt.show()
for i,f in enumerate(f_ds):
    if i%2000==0:
        plt.imshow(f,cmap='gray')
        plt.show()




        
    

