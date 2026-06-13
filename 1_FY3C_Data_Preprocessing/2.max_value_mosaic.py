import os
import glob
import numpy as np
from osgeo import gdal
from math import ceil
from tqdm import tqdm

gdal.UseExceptions()

def get_data_list(file_path, prix=""):
    """
    获取指定路径下的文件列表，可按前缀过滤
    
    Parameters:
    -----------
    file_path : str
        文件夹路径
    prix : str
        文件名过滤条件（包含指定字符串的文件）
    
    Returns:
    --------
    list : 文件完整路径列表
    """
    list1 = []
    if os.path.isdir(file_path):
        fileList = os.listdir(file_path)
        if prix != "":
            for f in fileList:
                if prix in f:
                    pre_data = os.path.join(file_path, f)
                    list1.append(pre_data)
        else:
            for f in fileList:
                pre_data = os.path.join(file_path, f)
                list1.append(pre_data)
    return list1

def get_same_image_list(infile_list, date_pos):
    """
    从文件列表中提取唯一的日期标识
    
    Parameters:
    -----------
    infile_list : list
        文件路径列表
    date_pos : int
        日期在文件名分割后的位置索引
    
    Returns:
    --------
    list : 唯一日期列表
    """
    image_list = []
    for file in infile_list:
        filename = os.path.basename(file).split('_')[date_pos]
        if filename not in image_list:
            image_list.append(filename)
    return list(set(image_list))

def get_same_list(image, infile_list):
    """
    获取包含指定日期标识的文件列表
    
    Parameters:
    -----------
    image : str
        日期标识
    infile_list : list
        文件路径列表
    
    Returns:
    --------
    list : 包含指定日期的文件列表
    """
    infile_list02 = []
    for data in infile_list:
        if image in data:
            infile_list02.append(data)
    return infile_list02

def GetExtent(infile):
    """获取栅格文件的地理范围"""
    ds = gdal.Open(infile)
    geotrans = ds.GetGeoTransform()
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    min_x, max_y = geotrans[0], geotrans[3]
    max_x, min_y = geotrans[0] + xsize * geotrans[1], geotrans[3] + ysize * geotrans[5]
    ds = None
    return min_x, max_y, max_x, min_y

def get_gdal_datatype(input_datatype):
    """
    将输入的数据类型转换为有效的GDAL数据类型
    """
    # 数据类型映射表
    datatype_mapping = {
        1: gdal.GDT_Byte,      # 8位无符号整型
        2: gdal.GDT_UInt16,    # 16位无符号整型  
        3: gdal.GDT_Int16,     # 16位有符号整型
        4: gdal.GDT_UInt32,    # 32位无符号整型
        5: gdal.GDT_Int32,     # 32位有符号整型
        6: gdal.GDT_Float32,   # 32位浮点型
        7: gdal.GDT_Float64,   # 64位浮点型
        8: gdal.GDT_CInt16,    # 16位复数
        9: gdal.GDT_CInt32,    # 32位复数
        10: gdal.GDT_CFloat32, # 32位复数浮点
        11: gdal.GDT_CFloat64  # 64位复数浮点
    }
    
    # 如果输入是整数，尝试从映射表获取
    if isinstance(input_datatype, int):
        return datatype_mapping.get(input_datatype, gdal.GDT_Float32)
    
    # 如果输入已经是GDAL数据类型，直接返回
    return input_datatype

def RasterMosaicAdvanced(file_list, outpath, method='max', nodata_value=None):
    """
    高级栅格镶嵌函数，支持多种融合方式
    
    Parameters:
    -----------
    file_list : list
        输入栅格文件列表
    outpath : str
        输出路径
    method : str
        融合方式，可选：'max', 'mean', 'min', 'first', 'last'
    nodata_value : float
        无效值，None表示自动检测
    """
    
    if not file_list:
        print("文件列表为空，跳过处理")
        return
    
    # 检查输出文件是否已存在
    if os.path.exists(outpath):
        print(f"{outpath} 已存在，跳过处理")
        return
    
    print(f"开始使用 {method.upper()} 方法进行栅格镶嵌...")
    print(f"输入文件数量: {len(file_list)}")
    
    Open = gdal.Open
    
    # 计算总的边界范围
    min_x, max_y, max_x, min_y = GetExtent(file_list[0])
    for infile in file_list:
        minx, maxy, maxx, miny = GetExtent(infile)
        min_x, min_y = min(min_x, minx), min(min_y, miny)
        max_x, max_y = max(max_x, maxx), max(max_y, maxy)
    
    # 获取参考信息
    in_ds = Open(file_list[0])
    in_band = in_ds.GetRasterBand(1)
    geotrans = list(in_ds.GetGeoTransform())
    width, height = geotrans[1], geotrans[5]
    columns = ceil((max_x - min_x) / width)
    rows = ceil((max_y - min_y) / (-height))
    
    # 获取波段数量和数据类型
    band_count = in_ds.RasterCount
    data_type = get_gdal_datatype(in_band.DataType)
    
    # 创建输出数据集
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(outpath, columns, rows, band_count, data_type, 
                          options=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=YES"])
    out_ds.SetProjection(in_ds.GetProjection())
    geotrans[0] = min_x
    geotrans[3] = max_y
    out_ds.SetGeoTransform(geotrans)
    inv_geotrans = gdal.InvGeoTransform(geotrans)
    
    # 自动检测NoData值
    if nodata_value is None:
        nodata_value = in_band.GetNoDataValue()
        if nodata_value is None:
            nodata_value = 0
    
    # 为平均值计算创建额外的数组来记录像素数量
    if method == 'mean':
        sum_arrays = [np.zeros((rows, columns), dtype=np.float64) for _ in range(band_count)]
        count_arrays = [np.zeros((rows, columns), dtype=np.int32) for _ in range(band_count)]
    else:
        # 为其他方法初始化输出数组
        output_arrays = []
        for i in range(band_count):
            if method == 'max':
                init_value = -np.inf
            elif method == 'min':
                init_value = np.inf
            else:  # first, last
                init_value = nodata_value
            
            arr = np.full((rows, columns), init_value, dtype=np.float64)
            output_arrays.append(arr)
    
    in_ds = None
    
    # 处理每个输入文件
    for file_idx, in_fn in enumerate(tqdm(file_list, desc=f"处理影像({method})")):
        in_ds = Open(in_fn)
        in_gt = in_ds.GetGeoTransform()
        offset = gdal.ApplyGeoTransform(inv_geotrans, in_gt[0], in_gt[3])
        x, y = map(int, offset)
        
        for i in range(band_count):
            # 读取当前影像数据
            current_data = in_ds.GetRasterBand(i+1).ReadAsArray()
            if current_data is None:
                continue
                
            current_data = current_data.astype(np.float64)
            h, w = current_data.shape
            
            # 计算有效的写入范围
            y_start = max(0, y)
            x_start = max(0, x)
            y_end = min(y + h, rows)
            x_end = min(x + w, columns)
            
            if y_end <= y_start or x_end <= x_start:
                continue
            
            # 计算在原始数据中的对应范围
            data_y_start = y_start - y
            data_x_start = x_start - x
            data_y_end = data_y_start + (y_end - y_start)
            data_x_end = data_x_start + (x_end - x_start)
            
            # 提取有效数据区域
            data_slice = current_data[data_y_start:data_y_end, data_x_start:data_x_end]
            
            # 创建有效数据掩膜
            valid_mask = (data_slice != nodata_value) & (~np.isnan(data_slice))
            
            if method == 'mean':
                # 平均值方法：累加数值和计数
                sum_arrays[i][y_start:y_end, x_start:x_end] += np.where(valid_mask, data_slice, 0)
                count_arrays[i][y_start:y_end, x_start:x_end] += valid_mask.astype(np.int32)
                
            elif method == 'max':
                # 最大值方法
                current_output = output_arrays[i][y_start:y_end, x_start:x_end]
                output_arrays[i][y_start:y_end, x_start:x_end] = np.where(
                    valid_mask, 
                    np.maximum(current_output, data_slice),
                    current_output
                )
                
            elif method == 'min':
                # 最小值方法
                current_output = output_arrays[i][y_start:y_end, x_start:x_end]
                output_arrays[i][y_start:y_end, x_start:x_end] = np.where(
                    valid_mask,
                    np.minimum(current_output, data_slice),
                    current_output
                )
                
            elif method == 'first':
                # 第一个有效值方法
                current_output = output_arrays[i][y_start:y_end, x_start:x_end]
                first_time_mask = (current_output == nodata_value) & valid_mask
                output_arrays[i][y_start:y_end, x_start:x_end] = np.where(
                    first_time_mask,
                    data_slice,
                    current_output
                )
                
            elif method == 'last':
                # 最后一个有效值方法（覆盖）
                current_output = output_arrays[i][y_start:y_end, x_start:x_end]
                output_arrays[i][y_start:y_end, x_start:x_end] = np.where(
                    valid_mask,
                    data_slice,
                    current_output
                )
        
        in_ds = None
    
    # 处理最终结果并写入
    print("正在写入最终结果...")
    for i in range(band_count):
        if method == 'mean':
            # 计算平均值
            valid_count_mask = count_arrays[i] > 0
            final_array = np.where(
                valid_count_mask,
                sum_arrays[i] / count_arrays[i],
                nodata_value
            )
        else:
            final_array = output_arrays[i]
            # 处理未被赋值的像素
            if method in ['max', 'min']:
                unassigned_mask = np.isinf(final_array)
                final_array = np.where(unassigned_mask, nodata_value, final_array)
        
        # 写入输出数据集
        out_band = out_ds.GetRasterBand(i+1)
        out_band.WriteArray(final_array.astype(gdal_array.GDALTypeCodeToNumericTypeCode(data_type)))
        out_band.SetNoDataValue(nodata_value)
    
    out_ds = None
    print(f"{method.upper()} 方法拼接完成，结果保存至: {outpath}")

def RasterMosaicMemoryEfficient(file_list, outpath, method='max', nodata_value=None, 
                               tile_size=1024):
    """
    内存高效版本的栅格镶嵌，适合处理大数据集
    """
    
    if not file_list:
        print("文件列表为空，跳过处理")
        return
    
    # 检查输出文件是否已存在
    if os.path.exists(outpath):
        print(f"{outpath} 已存在，跳过处理")
        return
    
    print(f"开始使用 {method.upper()} 方法进行内存高效栅格镶嵌...")
    print(f"输入文件数量: {len(file_list)}")
    
    Open = gdal.Open
    
    # 计算总的边界范围
    min_x, max_y, max_x, min_y = GetExtent(file_list[0])
    for infile in file_list:
        minx, maxy, maxx, miny = GetExtent(infile)
        min_x, min_y = min(min_x, minx), min(min_y, miny)
        max_x, max_y = max(max_x, maxx), max(max_y, maxy)
    
    # 获取参考信息
    in_ds = Open(file_list[0])
    in_band = in_ds.GetRasterBand(1)
    geotrans = list(in_ds.GetGeoTransform())
    width, height = geotrans[1], geotrans[5]
    columns = ceil((max_x - min_x) / width)
    rows = ceil((max_y - min_y) / (-height))
    band_count = in_ds.RasterCount
    
    # 获取正确的数据类型
    data_type = get_gdal_datatype(in_band.DataType)
    
    # 创建输出数据集
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(outpath, columns, rows, band_count, data_type, 
                          options=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=YES"])
    out_ds.SetProjection(in_ds.GetProjection())
    geotrans[0] = min_x
    geotrans[3] = max_y
    out_ds.SetGeoTransform(geotrans)
    inv_geotrans = gdal.InvGeoTransform(geotrans)
    
    # 自动检测NoData值
    if nodata_value is None:
        nodata_value = in_band.GetNoDataValue()
        if nodata_value is None:
            nodata_value = 0
    
    # 设置输出波段的NoData值
    for i in range(band_count):
        out_ds.GetRasterBand(i+1).SetNoDataValue(nodata_value)
    
    in_ds = None
    
    # 分块处理
    for tile_y in tqdm(range(0, rows, tile_size), desc="处理分块"):
        for tile_x in range(0, columns, tile_size):
            # 计算当前分块的大小
            current_tile_height = min(tile_size, rows - tile_y)
            current_tile_width = min(tile_size, columns - tile_x)
            
            # 为当前分块创建处理数组
            if method == 'mean':
                sum_arrays = [np.zeros((current_tile_height, current_tile_width), dtype=np.float64) 
                             for _ in range(band_count)]
                count_arrays = [np.zeros((current_tile_height, current_tile_width), dtype=np.int32) 
                               for _ in range(band_count)]
            else:
                if method == 'max':
                    init_value = -np.inf
                elif method == 'min':
                    init_value = np.inf
                else:  # first, last
                    init_value = nodata_value
                
                tile_arrays = [np.full((current_tile_height, current_tile_width), 
                                      init_value, dtype=np.float64) 
                              for _ in range(band_count)]
            
            # 处理每个输入文件对当前分块的贡献
            for in_fn in file_list:
                in_ds = Open(in_fn)
                in_gt = in_ds.GetGeoTransform()
                
                # 计算输入文件在输出坐标系中的位置
                offset = gdal.ApplyGeoTransform(inv_geotrans, in_gt[0], in_gt[3])
                file_x, file_y = map(int, offset)
                
                # 检查是否与当前分块重叠
                if (file_x >= tile_x + current_tile_width or 
                    file_y >= tile_y + current_tile_height or
                    file_x + in_ds.RasterXSize <= tile_x or 
                    file_y + in_ds.RasterYSize <= tile_y):
                    in_ds = None
                    continue
                
                # 计算重叠区域
                overlap_x_start = max(tile_x, file_x)
                overlap_y_start = max(tile_y, file_y)
                overlap_x_end = min(tile_x + current_tile_width, file_x + in_ds.RasterXSize)
                overlap_y_end = min(tile_y + current_tile_height, file_y + in_ds.RasterYSize)
                
                # 在分块数组中的位置
                tile_x_start = overlap_x_start - tile_x
                tile_y_start = overlap_y_start - tile_y
                tile_x_end = overlap_x_end - tile_x
                tile_y_end = overlap_y_end - tile_y
                
                # 在输入文件中的位置
                file_x_start = overlap_x_start - file_x
                file_y_start = overlap_y_start - file_y
                file_x_end = overlap_x_end - file_x
                file_y_end = overlap_y_end - file_y
                
                # 处理每个波段
                for i in range(band_count):
                    data = in_ds.GetRasterBand(i+1).ReadAsArray(
                        file_x_start, file_y_start, 
                        file_x_end - file_x_start, file_y_end - file_y_start
                    )
                    
                    if data is None:
                        continue
                    
                    data = data.astype(np.float64)
                    valid_mask = (data != nodata_value) & (~np.isnan(data))
                    
                    if method == 'mean':
                        sum_arrays[i][tile_y_start:tile_y_end, tile_x_start:tile_x_end] += np.where(valid_mask, data, 0)
                        count_arrays[i][tile_y_start:tile_y_end, tile_x_start:tile_x_end] += valid_mask.astype(np.int32)
                    
                    elif method == 'max':
                        current_tile = tile_arrays[i][tile_y_start:tile_y_end, tile_x_start:tile_x_end]
                        tile_arrays[i][tile_y_start:tile_y_end, tile_x_start:tile_x_end] = np.where(
                            valid_mask, np.maximum(current_tile, data), current_tile
                        )
                    
                    elif method == 'min':
                        current_tile = tile_arrays[i][tile_y_start:tile_y_end, tile_x_start:tile_x_end]
                        tile_arrays[i][tile_y_start:tile_y_end, tile_x_start:tile_x_end] = np.where(
                            valid_mask, np.minimum(current_tile, data), current_tile
                        )
                    
                    elif method == 'first':
                        current_tile = tile_arrays[i][tile_y_start:tile_y_end, tile_x_start:tile_x_end]
                        first_time_mask = (current_tile == nodata_value) & valid_mask
                        tile_arrays[i][tile_y_start:tile_y_end, tile_x_start:tile_x_end] = np.where(
                            first_time_mask, data, current_tile
                        )
                    
                    elif method == 'last':
                        current_tile = tile_arrays[i][tile_y_start:tile_y_end, tile_x_start:tile_x_end]
                        tile_arrays[i][tile_y_start:tile_y_end, tile_x_start:tile_x_end] = np.where(
                            valid_mask, data, current_tile
                        )
                
                in_ds = None
            
            # 写入当前分块的结果
            for i in range(band_count):
                if method == 'mean':
                    valid_count_mask = count_arrays[i] > 0
                    final_tile = np.where(
                        valid_count_mask,
                        sum_arrays[i] / count_arrays[i],
                        nodata_value
                    )
                else:
                    final_tile = tile_arrays[i]
                    if method in ['max', 'min']:
                        unassigned_mask = np.isinf(final_tile)
                        final_tile = np.where(unassigned_mask, nodata_value, final_tile)
                
                # 根据数据类型转换
                if data_type == gdal.GDT_Float32:
                    final_tile = final_tile.astype(np.float32)
                elif data_type == gdal.GDT_Float64:
                    final_tile = final_tile.astype(np.float64)
                elif data_type == gdal.GDT_Int16:
                    final_tile = final_tile.astype(np.int16)
                elif data_type == gdal.GDT_UInt16:
                    final_tile = final_tile.astype(np.uint16)
                elif data_type == gdal.GDT_Byte:
                    final_tile = final_tile.astype(np.uint8)
                else:
                    final_tile = final_tile.astype(np.float32)
                
                out_ds.GetRasterBand(i+1).WriteArray(final_tile, tile_x, tile_y)
    
    out_ds = None
    print(f"{method.upper()} 方法内存高效拼接完成，结果保存至: {outpath}")

def process_daily_mosaic(infile_path, outfile_path, prix="_NDVI.tif", date_pos=0, 
                        method='max', memory_efficient=True, tile_size=1024):
    """
    按日期分组进行栅格镶嵌的主函数
    
    Parameters:
    -----------
    infile_path : str
        输入文件夹路径
    outfile_path : str
        输出文件夹路径
    prix : str
        文件名过滤条件（如 "_NDVI.tif"）
    date_pos : int
        日期在文件名分割后的位置索引
    method : str
        融合方式：'max', 'mean', 'min', 'first', 'last'
    memory_efficient : bool
        是否使用内存高效版本
    tile_size : int
        分块大小（仅内存高效版本使用）
    """
    
    # 确保输出目录存在
    os.makedirs(outfile_path, exist_ok=True)
    
    # 获取所有符合条件的文件
    print(f"正在扫描文件夹: {infile_path}")
    print(f"文件过滤条件: {prix}")
    
    infile_list = get_data_list(infile_path, prix=prix)
    
    if not infile_list:
        print("未找到符合条件的文件！")
        return
    
    print(f"找到 {len(infile_list)} 个文件")
    
    # 获取所有唯一的日期标识
    image_name_list = get_same_image_list(infile_list, date_pos)
    image_name_list.sort()
    
    print(f"找到 {len(image_name_list)} 个不同的日期:")
    for name in image_name_list[:5]:  # 显示前5个日期
        print(f"  - {name}")
    if len(image_name_list) > 5:
        print(f"  ... 还有 {len(image_name_list) - 5} 个日期")
    
    # 处理每个日期的数据
    failed_dates = []
    successful_dates = []
    
    for i, name in enumerate(image_name_list):
        print(f"\n=== 处理第 {i+1}/{len(image_name_list)} 个日期: {name} ===")
        
        # 获取该日期的所有文件
        infile_list02 = get_same_list(name, infile_list)
        print(f"该日期包含 {len(infile_list02)} 个文件")
        
        # 构造输出文件名
        output_filename = f"{name}{prix}"
        output_path = os.path.join(outfile_path, output_filename)
        
        try:
            # 根据参数选择处理方法
            if memory_efficient or len(infile_list02) > 10 or method == 'mean':
                RasterMosaicMemoryEfficient(infile_list02, output_path, 
                                          method=method, tile_size=tile_size)
            else:
                RasterMosaicAdvanced(infile_list02, output_path, method=method)
            
            successful_dates.append(name)
            print(f"✓ 完成: {output_path}")
            
        except Exception as e:
            print(f"✗ 处理失败 {name}: {str(e)}")
            failed_dates.append(name)
    
    # 输出处理结果统计
    print(f"\n{'='*50}")
    print(f"处理完成！")
    print(f"成功处理: {len(successful_dates)} 个日期")
    print(f"失败处理: {len(failed_dates)} 个日期")
    
    if failed_dates:
        print(f"失败的日期: {failed_dates}")
    
    print(f"输出文件夹: {outfile_path}")
    print(f"融合方法: {method.upper()}")

if __name__ == '__main__':
    # 配置参数
    infile_path = r"I:\NDVI_10days_wgs84"        # 输入文件夹路径
    outfile_path = r"I:\NDVI_Final_wgs84"        # 输出文件夹路径
    prix = "_NDVI.tif"                           # 文件名过滤条件
    date_pos = 0                                 # 日期在文件名分割后的位置索引
    
    # 镶嵌参数
    method = 'max'                               # 融合方式：'max', 'mean', 'min', 'first', 'last'
    memory_efficient = True                      # 是否使用内存高效版本
    tile_size = 1024                            # 分块大小
    
    print("栅格按日期分组镶嵌程序")
    print("="*50)
    print(f"输入路径: {infile_path}")
    print(f"输出路径: {outfile_path}")
    print(f"文件过滤: {prix}")
    print(f"日期位置: 第{date_pos}个分割部分")
    print(f"融合方法: {method.upper()}")
    print(f"内存模式: {'高效模式' if memory_efficient else '标准模式'}")
    
    # 开始处理
    process_daily_mosaic(
        infile_path=infile_path,
        outfile_path=outfile_path,
        prix=prix,
        date_pos=date_pos,
        method=method,
        memory_efficient=memory_efficient,
        tile_size=tile_size
    )