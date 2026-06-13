import os
from osgeo import gdal
import h5py

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
    if type(hdf_ds.attrs['Left-Top X']) is np.ndarray:
        left_x = hdf_ds.attrs['Left-Top X'][0]
    else:
        left_x = float(hdf_ds.attrs['Left-Top X'])

    if type(hdf_ds.attrs['Left-Top Y']) is np.ndarray:
        left_y = hdf_ds.attrs['Left-Top Y'][0]
    else:
        left_y = float(hdf_ds.attrs['Left-Top Y'])

    res_x = hdf_ds.attrs['Resolution X'][0]
    res_y = hdf_ds.attrs['Resolution Y'][0]
    # bandname = list(hdf_ds.keys())[6]  # 5是evi 6是ndvi
    # print(list(hdf_ds.keys()))
    bands = list(hdf_ds.keys())

    if bandname not in bands:
        print(in_file, 'The HDF Do Not Has this Band:', bandname)
        bandname = 'VIRR_1KM_LST'
        if bandname not in bands:
            return

    band_ds = hdf_ds[bandname]
    Slope = band_ds.attrs['Slope'][0]
    Intercept = band_ds.attrs['Intercept'][0]
    valid_range = band_ds.attrs['valid_range']
    fill_value = band_ds.attrs['FillValue'][0]

    rows = band_ds.shape[0]
    cols = band_ds.shape[1]
    data = band_ds[()]
    valid_index = np.where((data >= valid_range[0]) & (data <= valid_range[1]) & (data != fill_value))

    out_data = np.full((rows, cols), np.nan,dtype=np.float32)
    out_data[valid_index] = data[valid_index]* Slope + Intercept
    #print(np.max(out_data[valid_index]), np.min(out_data[valid_index]))

    driver = gdal.GetDriverByName("GTiff")
    outds = driver.Create(out_file, cols, rows, 1, gdal.GDT_Float32)
    outds.SetGeoTransform(
        (float(left_x) * 1000,  # 切记250m的分辨率需要除以4
         float(res_x) * 1000,
         0,
         float(left_y) * 1000,  # 切记250m的分辨率需要除以4
         0,
         -1 * float(res_y) * 1000)
    )
    proj = 'PROJCS["World_Hammer",GEOGCS["Unknown datum based upon the custom spheroid",DATUM["Not_specified_based_on_custom_spheroid",SPHEROID["Custom spheroid",6363961,0]],PRIMEM["Greenwich",0],UNIT["Degree",0.0174532925199433]],PROJECTION["Hammer_Aitoff"],PARAMETER["False_Easting",0],PARAMETER["False_Northing",0],PARAMETER["Central_Meridian",0],UNIT["metre",1],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
    outds.SetProjection(proj)
    outband = outds.GetRasterBand(1)
    outband.WriteArray(out_data)
    #print(in_file, bandname, 'is ok!')


if __name__ == '__main__':
    for m in range(1,13):
        month = str(m)
        infile = os.path.join('I:\FY3C_VIRR_Global_COT_云光学厚度',month)
        outfile = r"I:\FY3C_2019\VIRR_COT"
        bands_name = ['Global CLoud Optical Thicknesss']#'1000M_10day_NDVI']#'VIRR_NDVI']#,'VIRR_NDVI']

        for bandname in bands_name:
            prix = bandname.split('_')[-1]
            infile_list = get_data_list(infile)
            outfile_list = get_data_list(infile, outfile, prix)
            for in_file, out_file in zip(infile_list, outfile_list):
                try:
                    HDF2Tif(in_file, out_file, bandname)
                    print(f"Finished Process:month-{month},{in_file}", bandname)
                    # print(in_file)
                except Exception as e:
                    print("except:", e)
                    print(in_file, bandname, 'is ERROR! Not Processed!')

           # print(f"Finished Process:{month}", bandname)
