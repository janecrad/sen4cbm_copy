#importing required packages
from skimage import data, io, filters
import skimage.io as skio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#Please write the directory containing the input files for:
#    1- tif file containing all bands (imageExample_Bands.tif here)
#    2- tif file contains the landuse classification (imageExample_SCL.tif here)
img_bands         = skio.imread(".\imageExample_Bands.tif")
img_mask    = skio.imread(".\imageExample_SCL.tif", plugin="tifffile")


#Main function to extract (mask) cloud, non-vegetation and unclassified codes from the remaining pixels
#This function ask for:
#                     the main image_filename contains different bands,
#                     the scl_mask_file contains the code for different landuse
#                     the layer_number (band number)
def apply_cloud_mask(image_filename, scl_mask_filename,layer_number):

    layer1 = image_filename[:,:,layer_number]

    Dim = np.shape(layer1)
    
    the_mask = np.eye(Dim[0], Dim[1])

    for i in range(0,Dim[0]):
        for j in range(0,Dim[1]):
            the_mask[i,j] = scl_mask_filename[i,j] == 4 or scl_mask_filename[i,j] == 5 or scl_mask_filename[i,j] == 7
        
    imstack1_1 = the_mask*layer1

    return imstack1_1


Dim = np.shape(img_bands)

#Loop for calculating the masked array for each band and saving the results as an image
##for band in range(0,Dim[2]):
##    data1 = apply_cloud_mask(img_bands, img_mask,band)
##    print(data1.mean())
##    im = Image.fromarray(data1)
##    out_name= 'Image_B' + str(band+1)+'.tif'
##    im.save(out_name)

##plt.imshow(data1)
##plt.show()

#---------------------------------------------------------------
#Saving the results as geotiff files using the following function

import matplotlib.pyplot as plt
import pandas as pd
import os.path
import re

from osgeo import gdal
from osgeo import gdal_array
from osgeo import osr

def array2raster(newRasterfn, dataset, array, dtype):
    """
    save GTiff file from numpy.array
    input:
        newRasterfn: save file name
        dataset : original tif file
        array : numpy.array
        dtype: Byte or Float32.
    """
    cols = array.shape[1]
    rows = array.shape[0]
    originX, pixelWidth, b, originY, d, pixelHeight = dataset.GetGeoTransform() 

    driver = gdal.GetDriverByName('GTiff')

    # set data type to save.
    GDT_dtype = gdal.GDT_Unknown
    if dtype == "Byte": 
        GDT_dtype = gdal.GDT_Byte
    elif dtype == "Float32":
        GDT_dtype = gdal.GDT_Float32
    else:
        print("Not supported data type.")

    # set number of band.
    if array.ndim == 2:
        band_num = 1
    else:
        band_num = array.shape[2]

    outRaster = driver.Create(newRasterfn, cols, rows, band_num, GDT_dtype)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))

    # Loop over all bands.
    for b in range(band_num):
        outband = outRaster.GetRasterBand(b + 1)
        # Read in the band's data into the third dimension of our array
        if band_num == 1:
            outband.WriteArray(array)
        else:
            outband.WriteArray(array[:,:,b])

    # setteing srs from input tif file.
    prj=dataset.GetProjection()
    outRasterSRS = osr.SpatialReference(wkt=prj)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()

Dim = np.shape(img_bands)
ds1 = gdal.Open('.\imageExample_SCL.tif')
for band in range(0,Dim[2]):
    data1 = apply_cloud_mask(img_bands, img_mask,band)
    out_name= 'Image_Georef_B' + str(band+1)+'.tif'
    array2raster(out_name, ds1, data1, 'Float32')

##data1 = apply_cloud_mask(img_bands, img_mask,1)
##array2raster('test1.tif', ds1, data1, 'Float32')
