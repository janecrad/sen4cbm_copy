# Import GDAL, numpy and os
from osgeo import gdal
import numpy as np
import os


def apply_cloud_mask(image_filename, scl_mask_filename):
    """Compute the Cloud Mask Matrix

    Args:
        image_filename (str): filename of the Sentinel 2 tiff images with 12 bands
        scl_mask_filename (str): filename of the Scene Classification Layer (SLC)

    Returns:
        masked_image_matrix (list of numpy.ndarray): a list of masked arrays
    """

    # Setup of the replacement value and invalid observations labels:
    no_data = 0
    invalid_px = [1, 2, 3, 6, 8, 9, 10, 11]

    # Import of the Sentinel bands and SLC layer:
    img_ = gdal.Open(str(image_filename))
    scl_mask = gdal.Open(str(scl_mask_filename))

    # Conversion of the SLC to an array:
    scl_mask_band = scl_mask.GetRasterBand(1)
    scl_mask_arr = scl_mask_band.ReadAsArray()

    # List initialization:
    masked_image_matrix = []

    # Loop through bands:
    for band_ in range(img_.RasterCount):

        # Getting the right band and conversion to an array:
        img_band = img_.GetRasterBand(band_ + 1)
        arr_band = img_band.ReadAsArray()

        # Loop through invalid observations list:
        for inv_px in invalid_px:

            # Replace the invalid observation with the replacement value:
            arr_band[scl_mask_arr == inv_px] = no_data

        # Append the masked array to the list:
        masked_image_matrix.append(arr_band)

    return masked_image_matrix


def scale_arr(arr_):
    """Compute the Scaled Array with value range 0-255

    Args:
        arr_ (numpy.ndarray): an array with real observations

    Returns:
        arr_sc (numpy.ndarray): a scaled array
    """

    # Find the maximum value in the array (without nans):
    max_ = np.nanmax(arr_)

    # Scale the array, based on the maximum values for scaled in unscaled images
    arr_sc = (arr_ * 255) / max_

    return arr_sc


def create_masked_tiff(masked_image_matrix, image_filename, output_filename, bands_=None):
    """Create the tiff image in LZW compression and 8 bit unsigned int

    Args:
        masked_image_matrix (list of numpy.ndarray): a list of masked arrays
        image_filename (str): filename of the Sentinel 2 tiff images with 12 bands
        output_filename (str): filename for the new tiff image
        bands_ (list of int): a list of bands to create the tiff

    Returns:
        masked_img_lzw (geotiff): compressed geotiff image
    """

    # Check if the bands are included:
    if bands_ == None:

        # If not, create the tiff from all bands
        bands_ = np.arange(len(masked_image_matrix))

    # Get the transformation of the initial image:
    input_img = gdal.Open(image_filename)
    input_trans = input_img.GetGeoTransform()

    # Get number of rows and columns of the array:
    rows_, cols_ = masked_image_matrix[0].shape

    # Setup the GDAL driver to geotiff:
    driver_ = gdal.GetDriverByName("GTiff")

    # Create a temporary filename:
    output_filename_nlzw = output_filename[:-4] + '_temp.tif'

    # Create a temporary geotiff (not compressed):
    masked_img = driver_.Create(output_filename_nlzw, cols_, rows_, len(bands_), gdal.GDT_Byte)

    # Setup the transformation from the initial image:
    masked_img.SetGeoTransform(input_trans)

    # Loop through the length of the selected bands:
    for i in range(len(bands_)):

        # Scale the array to the range of the unsigned 8-bit intiger format:
        masked_image_arr_sc = scale_arr(arr_=masked_image_matrix[bands_[i]])

        # Get the right band of the temporary geotiff:
        masked_band = masked_img.GetRasterBand(i + 1)

        # Write the scaled array as a selected band:
        masked_band.WriteArray(masked_image_arr_sc.astype(np.uint8))

    # Flush cached arrays to disk:
    masked_img.FlushCache()

    # Get the compression type option as LZW:
    trans_opt = gdal.TranslateOptions(creationOptions=['COMPRESS=LZW'])

    # Geotiff compression:
    masked_img_lzw = gdal.Translate(output_filename, masked_img, options=trans_opt)

    # Remove temporary geotiff from the memory and disk
    del masked_img
    os.remove(output_filename_nlzw)

    return masked_img_lzw


