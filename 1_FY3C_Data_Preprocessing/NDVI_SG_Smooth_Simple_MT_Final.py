# -*- coding: utf-8 -*-
import os
import glob
import re
import datetime
import math
import numpy as np
from osgeo import gdal
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ==============================================================================
# --- 1. 用户配置区域 ---
# ==============================================================================

# --- 路径配置 ---
# 输入包含时间序列影像的文件夹
INPUT_DIR = '../VIRR_10days_NDVI'
# 输出平滑后影像的文件夹
OUTPUT_DIR = '../VIRR_10days_NDVI_GF'
# 区域掩膜文件 (ROI), 0为不处理区域, 1为处理区域
ROI_FILE = '../FY3_VIRR_LST_NDVI_LandBound_2019_1km_ROI.tif'  # 如果不需要掩膜，请设置为 None
# --- 性能配置 ---
BLOCK_SIZE = 512
MULTIPROCESSING = True
NUM_WORKERS = 0

# --- 调试与验证配置 ---
ENABLE_SENTRY_BLOCK_CHECK = True
SENTRY_BLOCK_COORDS = None

# --- 平滑与重建参数 ---
FILL_VALUE = -9999
SPLINE_POWER = 2
WIDE_WINDOW = 11
WIDE_POWER = 2
NARROW_WINDOW = 5
NARROW_POWER = 3


# ==============================================================================
# --- 2. 核心功能函数 ---
# ==============================================================================

def parse_date_from_filename(filename):
    """从文件名中解析日期 (YYYYMMDD)"""
    basename = os.path.basename(filename)
    match = re.search(r'(\d{8})', basename)
    if match:
        try:
            return datetime.datetime.strptime(match.group(1), '%Y%m%d').date()
        except ValueError:
            return None
    return None


def find_valid_sentry_block(roi_file, rows, cols, block_size):
    """智能查找一个包含有效数据(ROI值为1)的哨兵块。"""
    if not roi_file or not os.path.exists(roi_file):
        return (rows // block_size // 2, cols // block_size // 2)

    roi_ds = gdal.Open(roi_file)
    if not roi_ds:
        return None

    num_blocks_y = math.ceil(rows / block_size)
    num_blocks_x = math.ceil(cols / block_size)

    search_points = [(0.5, 0.5), (0.25, 0.25), (0.75, 0.75), (0.25, 0.75), (0.75, 0.25), (0.1, 0.1), (0.9, 0.9)]

    for r_ratio, c_ratio in search_points:
        r_idx = int(num_blocks_y * r_ratio)
        c_idx = int(num_blocks_x * c_ratio)
        r_off = r_idx * block_size
        c_off = c_idx * block_size
        r_size = min(block_size, rows - r_off)
        c_size = min(block_size, cols - c_off)
        try:
            roi_block = roi_ds.GetRasterBand(1).ReadAsArray(c_off, r_off, c_size, r_size)
            if np.any(roi_block == 1):
                roi_ds = None
                return (r_idx, c_idx)
        except Exception:
            continue
    roi_ds = None
    return None


def smooth_pixel_timeseries(pixel_values, dates_ordinal):
    """对单个像素的时间序列进行平滑和重建"""
    valid_mask = (pixel_values != FILL_VALUE) & np.isfinite(pixel_values)
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) < max(WIDE_POWER + 1, NARROW_POWER + 1):
        return np.full_like(pixel_values, FILL_VALUE)
    valid_vals = pixel_values[valid_indices]
    valid_dates = dates_ordinal[valid_indices]
    try:
        # ... (之前的插值和 Savitzky-Golay 滤波代码不变) ...
        interp_kind = 'linear' if len(valid_dates) <= SPLINE_POWER else SPLINE_POWER
        interp_func = interp1d(valid_dates, valid_vals, kind=interp_kind, bounds_error=False, fill_value=FILL_VALUE)
        full_series = interp_func(dates_ordinal)
        first_valid_date_ord, last_valid_date_ord = valid_dates[0], valid_dates[-1]
        # Extrapolate constant values before the first valid date and after the last valid date
        full_series[dates_ordinal < first_valid_date_ord] = valid_vals[0]
        full_series[dates_ordinal > last_valid_date_ord] = valid_vals[-1]

        # Ensure full_series is finite before applying Savitzky-Golay
        if not np.all(np.isfinite(full_series)):
             # If interpolation failed to produce a finite series, return FILL_VALUEs
             return np.full_like(pixel_values, FILL_VALUE)

        # Apply wide window Savitzky-Golay filter
        smoothed_series = savgol_filter(full_series, WIDE_WINDOW, WIDE_POWER, mode='nearest') # Added mode='nearest' for edge handling

        # Iteratively apply narrow window Savitzky-Golay and preserve original valid values
        for _ in range(3): # Number of iterations
            # Preserve original valid values (ensure smoothed series is >= original at valid points)
            # This operation is done IN-PLACE
            np.maximum(smoothed_series[valid_indices], valid_vals, out=smoothed_series[valid_indices])
            # Apply narrow window Savitzky-Golay filter
            smoothed_series = savgol_filter(smoothed_series, NARROW_WINDOW, NARROW_POWER, mode='nearest') # Added mode='nearest'

        # Ensure any potential NaNs introduced by filtering (unlikely with mode='nearest') are FILL_VALUE
        smoothed_series[~np.isfinite(smoothed_series)] = FILL_VALUE

        # --- 新增代码开始 ---
        # 检查并设置超出 NDVI 有效范围 [-1, 1] 的值 为 FILL_VALUE (NaN)
        # 使用布尔索引：创建一个掩膜，其中值为 True 的位置表示 smoothed_series 中的值小于 -1 或大于 1
        out_of_range_mask = (smoothed_series < -1) | (smoothed_series > 1)
        # 将这些超出范围位置的值设置为 FILL_VALUE
        smoothed_series[out_of_range_mask] = FILL_VALUE
        # --- 新增代码结束 ---

        return smoothed_series
    except Exception as e:
        # 捕获任何异常，返回填充值，避免程序崩溃
        print(f"处理像素时间序列时发生错误: {e}") # 可选：打印错误信息以便调试
        return np.full_like(pixel_values, FILL_VALUE)


def create_output_geotiffs(output_files, rows, cols, geotransform, projection):
    """预先创建所有输出文件"""
    driver = gdal.GetDriverByName("GTiff")
    for f in tqdm(output_files, desc="创建输出文件"):
        os.makedirs(os.path.dirname(f), exist_ok=True)
        out_ds = driver.Create(f, cols, rows, 1, gdal.GDT_Float32, options=['COMPRESS=LZW', 'TILED=YES'])
        if out_ds:
            out_ds.SetGeoTransform(geotransform)
            out_ds.SetProjection(projection)
            out_band = out_ds.GetRasterBand(1)
            out_band.SetNoDataValue(FILL_VALUE)
            out_ds.FlushCache()
            out_ds = None


def get_block_data(r_off, c_off, r_size, c_size, config):
    """辅助函数，仅用于为哨兵块加载原始数据"""
    num_images = len(config['input_files'])
    block_stack = np.full((num_images, r_size, c_size), FILL_VALUE, dtype=np.float32)
    for i, f in enumerate(config['input_files']):
        ds = gdal.Open(f)
        band = ds.GetRasterBand(1)
        data_block = band.ReadAsArray(c_off, r_off, c_size, r_size).astype(np.float32)
        data_block[np.isnan(data_block)] = FILL_VALUE
        nodata = band.GetNoDataValue()
        if nodata is not None:
            data_block[data_block == nodata] = FILL_VALUE
        block_stack[i, :, :] = data_block
        ds = None
    return block_stack


# ==============================================================================
# --- 3. 多进程工作函数 ---
# ==============================================================================

def process_block(args):
    """处理单个数据块的函数"""
    r_off, c_off, r_size, c_size, config = args

    if config['roi_file']:
        try:
            roi_ds = gdal.Open(config['roi_file'])
            roi_block = roi_ds.GetRasterBand(1).ReadAsArray(c_off, r_off, c_size, r_size)
            roi_ds = None
            if not np.any(roi_block == 1): return None
        except Exception:
            roi_block = np.ones((r_size, c_size), dtype=np.uint8)
    else:
        roi_block = np.ones((r_size, c_size), dtype=np.uint8)

    block_stack = get_block_data(r_off, c_off, r_size, c_size, config)

    smoothed_block_stack = np.full_like(block_stack, FILL_VALUE)
    process_indices = np.argwhere(roi_block == 1)

    for r_loc, c_loc in process_indices:
        pixel_timeseries = block_stack[:, r_loc, c_loc]
        if np.all(pixel_timeseries == FILL_VALUE): continue
        smoothed_series = smooth_pixel_timeseries(pixel_timeseries, config['dates_ordinal'])
        smoothed_block_stack[:, r_loc, c_loc] = smoothed_series

    return (r_off, c_off, smoothed_block_stack)


# ==============================================================================
# --- 4. 主处理流程 ---
# ==============================================================================

if __name__ == "__main__":
    print("--- NDVI时间序列平滑与重建开始 (带智能哨兵验证 v2) ---")

    # --- 步骤 1: 文件和元数据准备 ---
    all_files = sorted(glob.glob(os.path.join(INPUT_DIR, '*.tif')))
    if not all_files: raise FileNotFoundError(f"在 '{INPUT_DIR}' 中未找到任何 .tif 文件。")

    file_date_pairs = sorted([(d, f) for f in all_files if (d := parse_date_from_filename(f))])
    if not file_date_pairs: raise ValueError("无法从任何文件名中解析出日期。")

    dates, sorted_files = zip(*file_date_pairs)
    print(f"找到 {len(sorted_files)} 个有效的时间序列影像。")

    ds = gdal.Open(sorted_files[0])
    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    rows, cols = ds.RasterYSize, ds.RasterXSize
    ds = None

    output_files = [os.path.join(OUTPUT_DIR, os.path.basename(f)) for f in sorted_files]
    create_output_geotiffs(output_files, rows, cols, geotransform, projection)

    # --- 步骤 2: 准备任务列表和共享配置 ---
    tasks = []
    num_blocks_y = math.ceil(rows / BLOCK_SIZE)
    num_blocks_x = math.ceil(cols / BLOCK_SIZE)
    for r_idx in range(num_blocks_y):
        for c_idx in range(num_blocks_x):
            r_off = r_idx * BLOCK_SIZE
            c_off = c_idx * BLOCK_SIZE
            r_size = min(BLOCK_SIZE, rows - r_off)
            c_size = min(BLOCK_SIZE, cols - c_off)
            tasks.append(((r_idx, c_idx), (r_off, c_off, r_size, c_size)))

    start_date = dates[0]
    dates_ordinal = np.array([(d - start_date).days for d in dates])

    config = {
        'input_files': sorted_files,
        'roi_file': ROI_FILE,
        'dates_ordinal': dates_ordinal,
    }

    # --- 步骤 3: 智能 "哨兵块" 验证 ---
    sentry_result = None  # 存储哨兵块处理结果
    sentry_task_to_remove = None
    if ENABLE_SENTRY_BLOCK_CHECK:
        print("\n--- 哨兵块验证模式已启用 ---")
        sentry_block_coords = None
        if SENTRY_BLOCK_COORDS:
            print(f"使用用户指定的哨兵块坐标: {SENTRY_BLOCK_COORDS}")
            sentry_block_coords = SENTRY_BLOCK_COORDS
        else:
            print("正在自动查找有效的哨兵块...")
            sentry_block_coords = find_valid_sentry_block(ROI_FILE, rows, cols, BLOCK_SIZE)

        if not sentry_block_coords:
            print("\n!! 错误: 无法自动找到任何包含有效数据的哨兵块。程序中止。 !!\n")
            exit()

        print(f"已选定哨兵块 (行块:{sentry_block_coords[0]}, 列块:{sentry_block_coords[1]}) 进行初步验证...")

        sentry_task_to_run = None
        for task in tasks:
            if task[0] == sentry_block_coords:
                sentry_task_to_run = task[1] + (config,)
                sentry_task_to_remove = task
                break

        if sentry_task_to_run:
            # 获取原始数据用于对比
            sentry_original_block = get_block_data(*sentry_task_to_run[:4], config)

            # 处理哨兵块
            sentry_result = process_block(sentry_task_to_run)

            if sentry_result is None:
                print("警告: 选定的哨兵块处理后无数据。")
            else:
                _, _, sentry_result_block = sentry_result
                valid_original_mask = (sentry_original_block != FILL_VALUE) & np.isfinite(sentry_original_block)
                valid_result_mask = (sentry_result_block != FILL_VALUE) & np.isfinite(sentry_result_block)

                if not np.any(valid_result_mask):
                    print("\n!! 错误：哨兵块处理结果全为无效值。程序已中止。 !!\n")
                    exit()
                else:
                    print("--- 哨兵块验证通过！---")
                    if np.any(valid_original_mask):
                        min_orig = np.min(sentry_original_block[valid_original_mask])
                        max_orig = np.max(sentry_original_block[valid_original_mask])
                        print(f"  - 原始数据范围 (有效值):  Min = {min_orig:.4f}, Max = {max_orig:.4f}")
                    else:
                        print("  - 原始数据范围 (有效值):  无有效原始数据，全部被重建。")

                    min_recon = np.min(sentry_result_block[valid_result_mask])
                    max_recon = np.max(sentry_result_block[valid_result_mask])
                    print(f"  - 重建后数据范围 (有效值): Min = {min_recon:.4f}, Max = {max_recon:.4f}")

                    print("哨兵块重建后各期最大值 (仅显示前5期):")
                    max_vals = np.max(sentry_result_block, axis=(1, 2), where=valid_result_mask, initial=FILL_VALUE)
                    for i, max_val in enumerate(max_vals[:36]):
                        print(f"  - 第 {i + 1} 期最大值: {max_val:.4f}")
                    print("...........................................")

            # 从任务列表中移除哨兵块任务（因为已经处理了）
            if sentry_task_to_remove:
                tasks.remove(sentry_task_to_remove)

    # --- 步骤 4: 执行完整处理 ---
    print("\n--- 开始正式处理所有数据块 ---")
    output_datasets = [gdal.Open(f, gdal.GA_Update) for f in output_files]

    try:
        def write_result(result):
            if result:
                r_off, c_off, smoothed_block = result
                smoothed_block[smoothed_block == FILL_VALUE] = np.nan
                for i, out_ds in enumerate(output_datasets):
                    out_ds.GetRasterBand(1).WriteArray(smoothed_block[i, :, :], c_off, r_off)

        # 如果有哨兵块结果，先写入
        if sentry_result is not None:
            print("正在写入哨兵块处理结果...")
            write_result(sentry_result)

        if not MULTIPROCESSING:
            print("模式: 单进程")
            pbar = tqdm(tasks, desc="分块处理")
            for _, task_params in pbar:
                result = process_block(task_params + (config,))
                write_result(result)
        else:
            workers = NUM_WORKERS if NUM_WORKERS > 0 else cpu_count()
            print(f"模式: 多进程 (使用 {workers} 个核心)")
            full_tasks = [task[1] + (config,) for task in tasks]

            with Pool(processes=workers) as pool:
                with tqdm(total=len(full_tasks), desc="并行处理块") as pbar:
                    for result in pool.imap_unordered(process_block, full_tasks):
                        write_result(result)
                        pbar.update(1)
    finally:
        print("处理完成，正在关闭所有文件...")
        for out_ds in output_datasets:
            out_ds.FlushCache()
            out_ds = None

    print("--- 全部处理完成！ ---")