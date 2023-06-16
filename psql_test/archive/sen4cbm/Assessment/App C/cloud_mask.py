# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 04:19:05 2022

@author: vrind
"""

#import necessary modules--------------------------

import os
import gdal
import numpy as np
from matplotlib import pyplot as plt


#open and display the scl file---------------------
sclfile = "C:/Users/vrind/Desktop/imageExample_SCL.tif"
imgfile = "C:/Users/vrind/Desktop/imageExample_Bands.tif"
(fileRoot, fileExt) = os.path.splitext(imgfile)
outFileName = fileRoot + "_mod" + fileExt

#read scl file
scl = gdal.Open(sclfile) 
a = scl.GetRasterBand(1)
mask = a.ReadAsArray()

#dimension check
print(scl.RasterCount) 
[cols, rows] = mask.shape
print([cols,rows])

#read image file
img = gdal.Open(imgfile)
print(img.RasterCount)
band_1 = img.GetRasterBand(5) # vegetation 
band_2 = img.GetRasterBand(6) # non-vegetation  
band_3 = img.GetRasterBand(8) # unclassified 
b1 = band_1.ReadAsArray()  
b2 = band_2.ReadAsArray()  
b3 = band_3.ReadAsArray()  
img_1 = np.dstack((b1, b2, b3))  
 
#condition check
img_out = np.where((mask==4) | (mask==5) | (mask==7),1,0)

# expanding the dimensions of the mask array (2D) to the same dimension of the image
def apply_cloud_mask(image_filename, scl_mask_filename):
# Your code here
    #image_filename = image_
    #scl_mask_filename = sclfile
    # Expanding the dimensions of the mask array (2D) to the same dimension of the image
    mask_exp = np.expand_dims(scl_mask_filename,axis=2)
    new_mask = np.concatenate((mask_exp,mask_exp,mask_exp),axis=2)
    #Passing Example_image through SCL_masked_image
    masked_image_matrix = image_filename * new_mask
    # Plot masked image
    plt.imshow(masked_image_matrix)
    return masked_image_matrix

filteredimage = apply_cloud_mask(img_1, img_out)
print(filteredimage.ndim)   
#print(maskedimagematrix1.ndim)
# Plotting the input, mask and mask_filtered images

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
rows, cols = 2, 2
plt.subplot(rows, cols, 1)
plt.imshow(img_1)
plt.title("Example_Image")
plt.subplot(rows, cols, 2)
plt.imshow(img_out)
plt.title("Generated_Masked")
plt.subplot(rows, cols, 3)
plt.imshow(filteredimage)
plt.title("Masked_Filter_Image")
plt.show()

#from PIL import Image
#img = Image.fromarray(filteredimage, 'RGB')
#img.show()

#save output
driver = gdal.GetDriverByName("GTiff")
outdata = driver.Create(outFileName, cols, rows, 3, gdal.GDT_Byte)
outdata.SetGeoTransform(scl.GetGeoTransform())
outdata.SetProjection(scl.GetProjection())
out1 = outdata.GetRasterBand(1)
out2 = outdata.GetRasterBand(2)
out3 = outdata.GetRasterBand(3)
outdband = np.dstack((out1, out2, out3))
outdata.WriteArray(filteredimage)
outdata.FlushCache()
outdata = None
band=None
ds=None
