import os
import h5py
import netCDF4
from osgeo import gdal, osr

gdal.UseExceptions()


def get_data_list(file_path, out="", prix=""):
    list1 = []  # 文件的完整路径
    if os.path.isdir(file_path):
        fileList = os.listdir(file_path)
        if out != "" and prix != "":
            for f in fileList:
                out_data = out + "\\" + f
                out_data = out_data.replace(".HDF", '_' + prix + '.tif')
                list1.append(out_data)
        else:
            for f in fileList:
                pre_data = file_path + '\\' + f  # 文件的完整路径
                list1.append(pre_data)
        return list1


import numpy as np


def HDF2Tif(in_file, out_file, bandname):
    hdf_ds = h5py.File(in_file, "r")
    bands = list(hdf_ds.keys())

    if bandname not in bands:
        print(in_file, 'The HDF Do Not Has this Band:', bandname)
        return

    band_ds = hdf_ds[bandname]
    Slope = band_ds.attrs['Slope'][0]
    Intercept = band_ds.attrs['Intercept'][0]
    valid_range = band_ds.attrs['valid_range']
    valid_range = [0, 32767]
    fill_value = band_ds.attrs['FillValue'][0]

    rows = band_ds.shape[0]
    cols = band_ds.shape[1]
    data = band_ds[()]

    nanindex = np.where((data < valid_range[0]) | (data > valid_range[1]) | (data == fill_value))
    data = data * Slope + Intercept
    data[nanindex] = np.NAN

    geotransform = (-17334193.54, 25067.53, 0, 7344784.83, 0, -25067.53)

    driver = gdal.GetDriverByName("GTiff")
    outds = driver.Create(out_file, cols, rows, 1, gdal.GDT_Float32)
    outds.SetGeoTransform(geotransform)

    # 构造projection
    srs = osr.SpatialReference()
    # srs.ImportFromEPSG(3410)  # 定义输出的坐标系为Global, Equal-Area(EPSG: 3410)
    # outds.SetProjection(srs.ExportToWkt())  # 给新建图层赋予投影信息

    proj = 'PROJCS["NSIDC EASE-Grid Global (deprecated)",GEOGCS["Unspecified datum based upon the International 1924 Authalic Sphere (deprecated)", DATUM["Not_specified_based_on_International_1924_Authalic_Sphere", SPHEROID["International 1924 Authalic Sphere",6371228,0,AUTHORITY["EPSG","7057"]],AUTHORITY["EPSG","6053"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4053"]],PROJECTION["Cylindrical_Equal_Area"],PARAMETER["standard_parallel_1",30],PARAMETER["central_meridian",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","3410"]]'
    outds.SetProjection(proj)
    # 数据写出
    outds.GetRasterBand(1).WriteArray(data)
    outds.FlushCache()  # 将数据写入硬盘


if __name__ == '__main__':
    infile = r"I:\FY3C_PMW_LST_原始数据\初始HDF\2020"
    outfile = r"I:\FY3C_PMW_LST_原始数据\初始TIFF"
    bands_name = ['Descending orbit LST']  # , 'VIRR_NDVI','Ascending orbit LST', ]

    for bandname in bands_name:
        prix = bandname.split('_')[-1]
        infile_list = get_data_list(infile)
        outfile_list = get_data_list(infile, outfile, prix)
        for in_file, out_file in zip(infile_list, outfile_list):

            HDF2Tif(in_file, out_file, bandname)

            try:
                HDF2Tif(in_file, out_file, bandname)
                # print(in_file)
            except Exception as e:
                print("except:", e)
                print(in_file, bandname, 'is ERROR! Not Processed!')

        print("Finished Process:", bandname)
