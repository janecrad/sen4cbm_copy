# Import functions
from mask import apply_cloud_mask, create_masked_tiff

# Setup parameters:
image_filename = './Data/imageExample_Bands.tif'
scl_mask_filename = './Data/imageExample_SCL.tif'
output_filename = './Data/imageExample_masked.tif'
output_filename_comp = './Data/imageExample_masked_comp.tif'
# Example bands for composition:
bands_ = [3, 4, 5]

# Compute the cloud mask matrix:
masked_image_matrix = apply_cloud_mask(image_filename=image_filename,
                                       scl_mask_filename=scl_mask_filename)

# Create tiff image from cloud mask matrix:
masked_img_lzw = create_masked_tiff(masked_image_matrix=masked_image_matrix,
                                    image_filename=image_filename,
                                    output_filename=output_filename)

# Create composition from cloud mask matrix and chosen bands:
masked_img_lzw_comp = create_masked_tiff(masked_image_matrix=masked_image_matrix,
                                         image_filename=image_filename,
                                         output_filename=output_filename_comp,
                                         bands_=bands_)
