# Cloud Mask Replacement

Cloud Mask Replacement assessment 

Author: Rafal Marciniak<br />
E-mail: marciniakr@outlook.com<br />
E-mail: rav.marcin.geodezja@gmail.com<br />
URL: https://github.com/ravmarcin/MT_assessment.git<br />


The Cloud Mask Replacement repository was developed as a assessment for interview.
The requirements was to develop two functions.<br />
The first one should applied the cloud mask to Sentinel 1 image, based on Scene Classification Layer (SCL).<br />
The second should create the geotiff image based on the masked matrix, with LZM compression type and 8 bit unsigned format.

## Installation

Install the libraries needed to run the cloud mask functions.

```bash
pip install -r ./requirements.txt
```

## Usage

```python
# Import functions
from mask import apply_cloud_mask, create_masked_tiff
```

```python
# Setup parameters:
image_filename = './Data/imageExample_Bands.tif'
scl_mask_filename = './Data/imageExample_SCL.tif'
output_filename = './Data/imageExample_masked.tif'
output_filename_comp = './Data/imageExample_masked_comp.tif'
# Example bands for composition:
bands_ = [3, 4, 5]
```

```python
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

```


