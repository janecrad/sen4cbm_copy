import os
import subprocess
import sys
from datetime import datetime
import math
from shapely.geometry import shape
from osgeo import gdal, gdal_array, ogr


def keyword_search(file, keywords, sep=':'):
    if isinstance(keywords, str):
        keywords = [keywords]
    outdic = dict()
    with open(file, 'r') as f:
        for line in f:
            for substr in keywords:
                if line.lower().find(substr.lower()) != -1:
                    out = line.strip('\n').split(sep=sep)
                    if len(out) != 2:
                        print('- separator character used in keyword: {}'.format(substr))
                    else:
                        outdic.update({out[0].strip(): out[1].strip()})
    if len(keywords) != len(outdic):
        notfound = [kwd for kwd in keywords if kwd not in outdic.keys()]
        print('- {} keyword not found'.format(notfound))
    return outdic


def extract_dom(dom, node_name):
    node = (dom.getElementsByTagName(node_name))[0]
    out = node.childNodes[0].data
    return out

# Utilities functions for dates
def from_date_to_doy(date):
    # date = raw_input("Enter date: ")  ## format is 02-02-2016
    adate = datetime.strptime(date, "%d-%m-%Y")
    day_of_year = adate.timetuple().tm_yday
    return day_of_year


# Utilities functions for coordinate points
def get_scene_center(lat, lon):
    zita = sum([math.sin(math.pi * float(x) / 180) for x in lon]) / len(lon)
    chi = sum([math.cos(math.pi * float(x) / 180) for x in lon]) / len(lon)
    c_lat = sum([float(x) for x in lat]) / len(lat)
    c_lon = 180 * math.atan2(zita, chi) / math.pi
    return [c_lat, c_lon]


def coord_to_wkt(lat, lon, point=True):
    if not isinstance(lat, list):
        lat = [lat]
    if not isinstance(lon, list):
        lon = [lon]
    coord_list = [[float(x), float(y)] for x, y in zip(lon, lat)]
    if point:
        o = {"coordinates": coord_list[0],
             "type": 'Point'}
    else:
        o = {"coordinates": [coord_list],
             "type": "Polygon"}
    geom = shape(o)

    return geom.wkt


# Utilities functions for rasters
def get_spatial_attributes(src_ds):
    geotransform = src_ds.GetGeoTransform()
    srs = src_ds.GetProjection()
    ncol = src_ds.RasterXSize
    nrow = src_ds.RasterYSize
    nbands = src_ds.RasterCount
    return geotransform, srs, ncol, nrow, nbands


def export_array_from_source(filename, out_array, src_ds=None, fmt='GTiff', ndv=0,
                             geotransform=None, srs=None, ncol=None, nrow=None, nbands=None):
    if src_ds is not None:
        geotransform, srs, ncol, nrow, nbands = get_spatial_attributes(src_ds)
    elif (geotransform is None) or (srs is None) or (ncol is None) or (nrow is None) or (nbands is None):
        sys.exit('Check input data: input missing')
    else:
        pass
    gdal_dtype = gdal_array.NumericTypeCodeToGDALTypeCode(out_array.dtype)
    driver = gdal.GetDriverByName(fmt)
    out_ds = driver.Create(filename, ncol, nrow, nbands, gdal_dtype)
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(srs)
    if nbands > 1:
        for i, image in enumerate(out_array, 1):
            # print(i, image)
            out_ds.GetRasterBand(i).WriteArray(image, 0, 0)
            out_ds.GetRasterBand(i).SetNoDataValue(ndv)
    else:
        out_ds.GetRasterBand(1).WriteArray(out_array, 0, 0)
        out_ds.GetRasterBand(1).SetNoDataValue(ndv)
    out_ds.FlushCache()
    return filename


def convert_image(image_file, out_file, out_format='GTiff', out_dtype='Float64'):
    keywds = keyword_search(os.path.join(os.getcwd(), 'aux_files/config_file.txt'), 'GDAL_DIR')
    GDAL_TRANSLATE = os.path.join(keywds['GDAL_DIR'], 'gdal_translate')
    try:
        src_ds = gdal.Open(str(image_file))
        gtiffDriver = gdal.GetDriverByName('MEM')
        tmp_ds = gtiffDriver.CreateCopy('', src_ds, 0)
        if out_format == 'GTiff':
            ext = '.tiff'
        elif out_format == 'JP2OpenJPEG':
            ext = '.j2p'
        else:
            ext = '.tiff'
        tmp_im = os.path.join(os.getcwd(), 'im_tmp_file' + ext)
        gdal.GetDriverByName(out_format).CreateCopy(tmp_im, tmp_ds, 0)
        subprocess.call([GDAL_TRANSLATE, "-ot", out_dtype, "-of", out_format, "-unscale", tmp_im, out_file])
        src_ds.FlushCache()
        tmp_ds.FlushCache()
        os.remove(tmp_im)
    except KeyError:
        return False
    return True


def reproject_vector(vector_path, epsg, outfile=None):
    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataset = driver.Open(vector_path)
    # from Layer
    layer = dataset.GetLayer()
    spatialRef = layer.GetSpatialRef()
    epsg_in = spatialRef.GetAuthorityCode("PROJCS")
    if str(epsg) != str(epsg_in):
        if outfile is None:
            outpath = os.path.dirname(vector_path)
            filename = os.path.basename(vector_path).split('.')[0] + '_{}.shp'.format(epsg)
            outfile = os.path.join(outpath, filename)
        subprocess.call(['ogr2ogr', '-t_srs', 'EPSG:{}'.format(epsg), outfile, vector_path])
    else:
        outfile = vector_path
    return outfile


def find_envelope(vector_path, epsg=None, driver_name='ESRI Shapefile'):
    if epsg is not None:
        vector_path = reproject_vector(vector_path, epsg)
    driver = ogr.GetDriverByName(driver_name)
    dataset = driver.Open(vector_path)
    # from Layer
    layer = dataset.GetLayer()
    return layer.GetExtent()


def extent2wktstr(extent):
    ULlon=extent[0]
    LRlon = extent[1]
    ULlat = extent[2]
    LRlat = extent[3]

    lat = [ULlat, ULlat, LRlat, LRlat]
    lon = [ULlon, LRlon, LRlon, ULlon]

    return coord_to_wkt(lat, lon, point=False)
