import os.path
from mssa.mssa import mSSA
import pandas as pd
import numpy as np
from osgeo import gdal, gdal_array
from tqdm import tqdm

# 禁用GDAL异常
gdal.UseExceptions()


class RWImage:
    """遥感影像读写类"""

    def readimg_allinfo(self, filename):
        """读取影像所有信息"""
        data = gdal_array.LoadFile(filename)
        dataset = gdal.Open(filename)
        proj = dataset.GetProjection()
        im_geotrans = dataset.GetGeoTransform()
        return data, proj, im_geotrans

    def readimg_onlydata(self, filename):
        """只读取影像数据"""
        data = gdal_array.LoadFile(filename)
        return data

    def readimg_bulk(self, filename, *bulk):
        """读取影像块数据"""
        data = gdal.Open(filename)
        im_width = data.RasterXSize
        im_height = data.RasterYSize

        if len(bulk) != 0:
            x0, y0, xpixels, ypixels = bulk
        else:
            x0, y0, xpixels, ypixels = 0, 0, im_width, im_height

        im_geotrans = data.GetGeoTransform()
        im_proj = data.GetProjection()
        im_data = data.ReadAsArray(x0, y0, xpixels, ypixels)
        del data
        return im_data, im_proj, im_geotrans

    def writeimg(self, filename, im_proj, im_geotrans, im_data):
        """写入影像文件"""
        # 判断数据类型
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        else:
            im_bands, (im_height, im_width) = 1, im_data.shape

        # 创建文件
        driver = gdal.GetDriverByName('GTiff')
        data = driver.Create(filename, im_width, im_height, im_bands, datatype)
        data.SetGeoTransform(im_geotrans)
        data.SetProjection(im_proj)

        if im_bands == 1:
            data.GetRasterBand(1).WriteArray(im_data)
        else:
            for i in range(im_bands):
                data.GetRasterBand(i + 1).WriteArray(im_data[i])
        del data


def create_nodata_image(shape, nodata_value=np.nan):
    """创建nodata影像"""
    return np.full(shape, nodata_value, dtype=np.float32)


def load_daily_images(input_dir, begin_date, end_date, nodata_value=np.nan):
    """逐天读取影像文件"""
    rt = RWImage()

    # 生成完整的日期序列
    begin = pd.to_datetime(begin_date)
    end = pd.to_datetime(end_date)
    full_dates = pd.date_range(start=begin, end=end, freq='D')
    dates_str = full_dates.strftime('%Y%m%d').tolist()

    print(f"日期范围: {begin_date} - {end_date}")
    print(f"总天数: {len(dates_str)}")
    print("=" * 60)

    # 读取第一个存在的文件以获取图像尺寸和投影信息
    first_valid_file = None
    proj = None
    geotrans = None
    rows, cols = None, None

    for date in dates_str:
        filename = os.path.join(input_dir, f'FY3C_MWRIX_GBAL_L2_LST_MLT_ESD_{date}_POAD_025KM_MS_Descending orbit LST.tif')
        if os.path.exists(filename):
            try:
                data, proj, geotrans = rt.readimg_allinfo(filename)
                if len(data.shape) == 2:
                    rows, cols = data.shape
                else:
                    rows, cols = data.shape[1], data.shape[2]
                first_valid_file = filename
                print(f"✓ 参考文件: {filename}")
                print(f"  图像尺寸: {rows} x {cols}")
                break
            except Exception as e:
                print(f"✗ 读取参考文件失败: {filename} - {e}")
                continue

    if first_valid_file is None:
        raise FileNotFoundError("未找到任何有效的影像文件")

    # 初始化数据数组
    pro_data = np.full((len(dates_str), rows, cols), nodata_value, dtype=np.float32)

    # 逐天读取数据
    print("=" * 60)
    print("开始逐天读取影像...")

    success_count = 0
    missing_count = 0
    error_count = 0

    progress_bar = tqdm(enumerate(dates_str), desc="读取影像", total=len(dates_str))

    for i, date in progress_bar:
        filename = os.path.join(input_dir,
                                f'FY3C_MWRIX_GBAL_L2_LST_MLT_ESD_{date}_POAD_025KM_MS_Descending orbit LST.tif')

        if os.path.exists(filename):
            try:
                data = rt.readimg_onlydata(filename)

                # 处理数据维度
                if len(data.shape) == 2:
                    pro_data[i, :, :] = data
                else:
                    pro_data[i, :, :] = data[0]  # 如果是多波段，取第一个波段

                success_count += 1
                progress_bar.set_postfix(status=f"✓ {date}", refresh=True)

            except Exception as e:
                error_count += 1
                progress_bar.set_postfix(status=f"✗ {date} - {str(e)[:30]}", refresh=True)
                # 保持nodata值

        else:
            missing_count += 1
            progress_bar.set_postfix(status=f"○ {date}", refresh=True)
            # 保持nodata值

    print("=" * 60)
    print(f"读取统计:")
    print(f"  成功读取: {success_count} 天")
    print(f"  文件缺失: {missing_count} 天")
    print(f"  读取失败: {error_count} 天")
    print(f"  总计: {len(dates_str)} 天")

    # 打印缺失天的详细信息
    if missing_count > 0:
        print(f"\n缺失天数详情:")
        missing_dates = []
        for i, date in enumerate(dates_str):
            filename = os.path.join(input_dir,
                                    f'FY3C_MWRIX_GBAL_L2_LST_MLT_ESD_{date}_POAD_025KM_MS_Descending orbit LST.tif')
            if not os.path.exists(filename):
                missing_dates.append(date)

        # 按行打印缺失日期，每行10个
        for i in range(0, len(missing_dates), 10):
            print(f"  {' '.join(missing_dates[i:i + 10])}")

    if error_count > 0:
        print(f"\n读取失败天数详情:")
        error_dates = []
        for i, date in enumerate(dates_str):
            filename = os.path.join(input_dir,
                                    f'FY3C_MWRIX_GBAL_L2_LST_MLT_ESD_{date}_POAD_025KM_MS_Descending orbit LST.tif')
            if os.path.exists(filename):
                try:
                    rt.readimg_onlydata(filename)
                except:
                    error_dates.append(date)

        # 按行打印错误日期，每行10个
        for i in range(0, len(error_dates), 10):
            print(f"  {' '.join(error_dates[i:i + 10])}")

    return pro_data, proj, geotrans, full_dates


def apply_ssa_gapfill(pro_data, nodata_value=np.nan):
    """应用SSA进行缺失值重建"""
    days, rows, cols = pro_data.shape
    pro_data_filled = np.empty([days, rows, cols], dtype=np.float32)

    # 计算每个像素的最大值，用于判断是否有有效数据
    max_data = np.nanmax(pro_data, axis=0)

    print("开始SSA缺失值重建...")

    valid_pixel_count = 0
    processed_pixel_count = 0

    for i in tqdm(range(rows), desc="处理行"):
        for j in range(cols):
            # 检查该像素是否有有效数据
            if np.isnan(max_data[i, j]) or max_data[i, j] == 0:
                # 如果整个时间序列都是无效值，直接填充nodata
                pro_data_filled[:, i, j] = nodata_value
                continue

            # 提取该像素的时间序列
            pixel_values = pro_data[:, i, j]

            # 检查是否有足够的有效数据进行SSA
            valid_count = np.sum(~np.isnan(pixel_values) & (pixel_values > 0))

            if valid_count < 5:  # 如果有效数据点太少，至少需要5个点
                pro_data_filled[:, i, j] = pixel_values
                continue

            processed_pixel_count += 1

            try:
                # 准备SSA输入数据
                data = pd.DataFrame({
                    'x': np.arange(days),
                    'y': pixel_values
                })

                # 应用SSA模型
                model = mSSA(fill_in_missing=True)
                model.update_model(data.loc[:days - 1, ['y']])
                df = model.predict('y', 0, days - 1)

                # 获取预测结果
                reconstructed = df['Mean Predictions'].values

                # 保留原始有效数据，只重建缺失值
                final_values = reconstructed.copy()
                valid_indices = np.where((pixel_values > 0) & ~np.isnan(pixel_values))
                final_values[valid_indices] = pixel_values[valid_indices]

                pro_data_filled[:, i, j] = final_values
                valid_pixel_count += 1

            except Exception as e:
                # 如果SSA失败，保留原始数据
                if processed_pixel_count <= 10:  # 只打印前10个错误
                    print(f"SSA处理失败 at ({i},{j}): {e}")
                pro_data_filled[:, i, j] = pixel_values

    print(f"SSA处理统计:")
    print(f"  成功处理像素: {valid_pixel_count}")
    print(f"  尝试处理像素: {processed_pixel_count}")

    return pro_data_filled


def main():
    """主函数"""
    # 配置参数
    input_dir = r'I:\FY3C_PMW_LST_原始数据\初始TIFF转经纬度'
    output_dir = r'I:\FY3C_2019\PMW_LST'

    begin_date = '20180918'
    end_date = '20200203'
    nodata_value = np.nan

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 默认投影信息（如果读取的文件没有投影信息）
    default_proj = ('PROJCS["NSIDC EASE-Grid Global (deprecated)",'
                    'GEOGCS["Unspecified datum based upon the International 1924 Authalic Sphere (deprecated)", '
                    'DATUM["Not_specified_based_on_International_1924_Authalic_Sphere", '
                    'SPHEROID["International 1924 Authalic Sphere",6371228,0,AUTHORITY["EPSG","7057"]],AUTHORITY["EPSG","6053"]],'
                    'PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
                    'AUTHORITY["EPSG","4053"]],PROJECTION["Cylindrical_Equal_Area"],PARAMETER["standard_parallel_1",30],'
                    'PARAMETER["central_meridian",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,'
                    'AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","3410"]]')

    default_geotrans = (-17334193.54, 25067.53, 0, 7344784.83, 0, -25067.53)

    try:
        # 1. 逐天读取影像数据
        print("=" * 60)
        print("步骤1: 逐天读取影像数据")
        pro_data, proj, geotrans, full_dates = load_daily_images(
            input_dir, begin_date, end_date, nodata_value
        )

        # 使用默认投影信息（如果需要）
        if proj is None:
            proj = default_proj
        if geotrans is None:
            geotrans = default_geotrans

        # 2. 应用SSA缺失值重建
        print("=" * 60)
        print("步骤2: SSA缺失值重建")
        pro_data_filled = apply_ssa_gapfill(pro_data, nodata_value)

        # 3. 保存结果
        print("=" * 60)
        print("步骤3: 保存结果")
        rt = RWImage()

        # 保存合并文件
        out_merge_file = os.path.join(
            output_dir,
            f'{begin_date}_{end_date}_FY3C_MWRIX_D_25km_Gapfilled_LST.tif'
        )
        rt.writeimg(out_merge_file, proj, geotrans, pro_data_filled)
        print(f"✓ 合并文件已保存: {out_merge_file}")

        # 保存逐日文件
        print("保存逐日文件...")
        dates_str = full_dates.strftime('%Y%m%d').tolist()

        for doy, date in enumerate(tqdm(dates_str, desc="保存逐日文件")):
            out_single_file = os.path.join(
                output_dir,
                f'FY3C_MWRIX_D_25km_{date}_Gapfilled_LST.tif'
            )
            rt.writeimg(out_single_file, proj, geotrans, pro_data_filled[doy, :, :])

        print("=" * 60)
        print("处理完成!")
        print(f"输出目录: {output_dir}")
        print(f"处理日期范围: {begin_date} - {end_date}")
        print(f"总天数: {len(dates_str)}")

    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()