import numpy as np
import numpy.ma as ma
import os
import sys
from osgeo import gdal,osr

def GetPredictorsBBox(datset):
    """Get the bounding box coordinates and SRS of the fine resolution predictors."""
    geoTF = datset.GetGeoTransform()
    MinX = geoTF[0]
    MinY = geoTF[3] + geoTF[5] * datset.RasterYSize
    MaxX = geoTF[0] + geoTF[1] * datset.RasterXSize
    MaxY = geoTF[3]

    proj = datset.GetProjection()
    SRS = osr.SpatialReference(wkt=proj)

    BBox = {"coords": (MinX, MinY, MaxX, MaxY), "SRS": SRS}
    return BBox

def WarpRaster(dst, dst_ndv, src, src_ndv, BBox, resampling, outfile):
    """For the predictors' BBox, warp the src raster to match the dst raster."""
    resampling_methods = {"average": 5, "lanczos":4, "cubspline": 3, "cubic":2, "bilinear":1, "nearest": 0}
    if resampling not in resampling_methods.keys():
        raise ValueError("Invalid resampling method.Please choose from: " + ", ".join(resampling_methods.keys()))

    warp_options = gdal.WarpOptions(
        format="GTiff",
        outputBounds=BBox["coords"],
        outputBoundsSRS=BBox["SRS"],
        srcSRS=osr.SpatialReference(wkt=src.GetProjection()),
        dstSRS=osr.SpatialReference(wkt=dst.GetProjection()),
        xRes=dst.GetGeoTransform()[1],
        yRes=abs(dst.GetGeoTransform()[5]),
        srcNodata=src_ndv,
        dstNodata=dst_ndv,
        resampleAlg=resampling_methods[resampling],
    )
    return gdal.Warp(outfile, src, options=warp_options)

if __name__ == '__main__':
    # 获取文件所在的绝对路径
    file_path = os.path.abspath(__file__)
    # 调用 os.path.dirname 三次获取上上上级目录
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))
    data_dir = os.path.join(parent_dir, 'Data')
    # 读取ROI和基础数据
    base_file = os.path.join(data_dir, 'Base\FY3_VIRR_LST_NDVI_LandBound_2019_25km_ROI.tif')
    base_file_dataset = gdal.Open(base_file)
    BBox = GetPredictorsBBox(base_file_dataset)

    # 目录设置
    dir = data_dir+ r'\PMW_LST'  # LST数据目录
    # 获取所有LST文件
    file_list = [os.path.join(dir, f) for f in os.listdir(dir)
                 if f.startswith('FY3C_MWRIX_D_25km') and f.endswith('_Gapfilled_LST.tif')]
    print(f"找到 {len(file_list)} 个PWM LST文件待处理")
    out_dir = os.path.join(data_dir, 'Sub_to_25km\PMW_LST_25km')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for file in file_list:
        # 打开LST文件
        LST_dataset = gdal.Open(file)
        # 提取文件名中的日期信息
        date = os.path.basename(file).split('_')[4]
        # 构建输出文件名
        outfile = os.path.join(out_dir, f'FY3C_PMW_{date}_25km_LST.tif')

        # 进行重投影
        WarpRaster(base_file_dataset, 0, LST_dataset, np.nan, BBox, "average", outfile)
        print(f"处理完成: {outfile}")




































