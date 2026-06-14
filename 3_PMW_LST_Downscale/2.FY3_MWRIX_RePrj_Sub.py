from osgeo import gdal
import os
import glob

gdal.UseExceptions()


def mask_gdal(inMaskData, filepath, outfile,isneed_clip=False):
    """
    GDAL掩膜提取
    :param inMaskData: 圈选范围的路径  shp文件可以换成tif文件
    :param filepath: 要掩膜的tif文件
    :param outfile: 掩膜好的tif文件
    :return:
    """
    dataset = gdal.Open(filepath)  # 打开遥感影像

    if isneed_clip:
        out_temp = outfile.replace('.tif', '_temp.tif')

        gdal.Warp(out_temp, dataset, dstSRS='EPSG:4326', xRes=0.25,
                  yRes=0.25)  # 重投影
        gdal.Warp(outfile, out_temp, cutlineDSName=inMaskData, cropToCutline=True)  # 按掩膜提取
        os.remove(out_temp)
        print(filepath + '-----掩膜成功')
    else:

        gdal.Warp(outfile, dataset, dstSRS='EPSG:4326', xRes=0.25,
                  yRes=0.25)  # 重投影
        print(filepath + '-----重投影成功')




# 先转为经纬度0.25分辨率，再裁剪，不然结果会没投影。NSIDC EASE-Grid Global (deprecated)比较特殊。
if __name__ == '__main__':
    filepath = r"I:\FY3C_PMW_LST_原始数据\初始TIFF"  # 要裁剪的tif文件所在的文件夹
    outfile = r"I:\FY3C_PMW_LST_原始数据\初始TIFF转经纬度"
    inMaskData = r"E:\Application\FY_LST_Fusion\Data\Output\FY3C_MWRIX\shp\Heihe_large_roi.shp"  # 圈选范围的路径

    isneed_clip = False

    os.chdir(filepath)
    names = glob.glob("*.tif")  # 读取文件
    for name in names:
        out = outfile + "\\" + name.split(".")[0] + ".tif"  # 按照圈选范围提取出的影像所存放的路径
        mask_gdal(inMaskData, name, out,isneed_clip)
