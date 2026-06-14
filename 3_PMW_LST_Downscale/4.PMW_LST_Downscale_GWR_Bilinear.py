# -*- coding: utf-8 -*-
import numpy as np
from osgeo import gdal, osr
import os
import glob
import cv2
import re
from datetime import datetime
from collections import defaultdict
import calendar
from tqdm import tqdm
from scipy import ndimage  # 必须安装 scipy

# ==============================================================================
# 0. 配置区域
# ==============================================================================

# 基础路径
BASE_DATA_DIR = r'示例数据集\Data'

# 输入路径
NDVI_DIR = os.path.join(BASE_DATA_DIR, 'VIRR_10days_NDVI_GF')
PMW_DIR = os.path.join(BASE_DATA_DIR, 'Sub_to_25km', 'CDF_Corrected_PMW_LST_25km')
# DEM 用于提供纹理 + 陆地掩膜 (海洋部分为 NaN)
DEM_FILE = os.path.join(BASE_DATA_DIR, 'Base', 'MERIT_DEM_1D.tif')

# 输出路径
OUT_DIR = os.path.join(BASE_DATA_DIR, 'PMW_LST_Downscale_GWR_1')

# [新增] SHAP分析中间数据保存路径
SHAP_DIR = os.path.join(BASE_DATA_DIR, 'GWR_Downscale_SHAP_Data')
if not os.path.exists(SHAP_DIR):
    os.makedirs(SHAP_DIR)


# ==============================================================================
# 1. 工具函数
# ==============================================================================

def get_ndvi_dekad_date(date_obj):
    year, month, day = date_obj.year, date_obj.month, date_obj.day
    if day <= 10:
        target_day = 10
    elif day <= 20:
        target_day = 20
    else:
        _, target_day = calendar.monthrange(year, month)
    return f"{year}{month:02d}{target_day:02d}"


def read_tif(tif_path):
    """读取 TIF，自动处理 NoData 为 NaN"""
    if not os.path.exists(tif_path): raise FileNotFoundError(f"文件未找到: {tif_path}")
    ds = gdal.Open(tif_path)
    w, h = ds.RasterXSize, ds.RasterYSize
    geo = ds.GetGeoTransform()
    proj = ds.GetProjection()
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    nodata = band.GetNoDataValue()

    # 转为 float 以支持 NaN
    arr = arr.astype(np.float32)

    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    return arr, w, h, geo, proj


def robust_fill_gaps(arr):
    """
    [强力填充] 使用距离变换填补任意大小的空洞
    保证计算过程无死角
    """
    if not np.isnan(arr).any():
        return arr

    valid_mask = ~np.isnan(arr)
    indices = ndimage.distance_transform_edt(~valid_mask, return_distances=False, return_indices=True)
    arr_filled = arr[tuple(indices)]

    return arr_filled


def save_tif_masked(arr, out_path, w, h, geo, proj, mask_arr):
    """保存结果，并应用掩膜"""
    # 只有 mask_arr 不为 NaN 的地方才保留数据
    final_arr = np.where(~np.isnan(mask_arr), arr, np.nan)

    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(out_path, w, h, 1, gdal.GDT_Float32)
    out_ds.SetGeoTransform(geo)
    out_ds.SetProjection(proj)
    out_ds.GetRasterBand(1).WriteArray(final_arr)
    out_ds.GetRasterBand(1).SetNoDataValue(np.nan)
    out_ds.FlushCache()
    del out_ds


# ==============================================================================
# 2. 降尺度逻辑 (带填充与SHAP数据拦截)
# ==============================================================================

def run_downscaling_gwr_filled(pmw_arr, ndvi_1km, dem_1km, w_1km, h_1km, date_str=None, shap_out_dir=None):
    """
    降尺度核心流程: 填补空洞 -> GWR回归 -> 返回结果 (新增保存SHAP中间数据)
    """

    # --- A. 预处理：填补所有空洞 ---
    # 为了保证回归不报错，这里必须填补所有空洞
    pmw_filled = robust_fill_gaps(pmw_arr)

    if ndvi_1km is not None:
        ndvi_filled = robust_fill_gaps(ndvi_1km)
    else:
        ndvi_filled = np.zeros((h_1km, w_1km), dtype=np.float32)

    dem_filled = robust_fill_gaps(dem_1km)

    # --- B. 准备低分辨率因子 (25km) ---
    h_low, w_low = pmw_filled.shape
    ndvi_low = cv2.resize(ndvi_filled, (w_low, h_low), interpolation=cv2.INTER_AREA)
    dem_low = cv2.resize(dem_filled, (w_low, h_low), interpolation=cv2.INTER_AREA)

    # --- C. 全局回归 (GWR 趋势) ---
    Y = pmw_filled.flatten()
    X1 = ndvi_low.flatten()
    X2 = dem_low.flatten()
    X = np.vstack([X1, X2, np.ones_like(X1)]).T

    # ==========================================================
    # [新增逻辑]：保存中间数据用于后续 SHAP 分析
    # ==========================================================
    if date_str and shap_out_dir:
        shap_file_path = os.path.join(shap_out_dir, f"SHAP_Data_Downscale_{date_str}.npz")
        if not os.path.exists(shap_file_path):
            # 以压缩格式保存目标变量(Y)和特征变量(X_NDVI, X_DEM)
            np.savez_compressed(shap_file_path, Y_PMW=Y, X_NDVI=X1, X_DEM=X2)
    # ==========================================================

    try:
        # 最小二乘法求解系数
        C, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        a, b, c = C[0], C[1], C[2]

        # 1. 计算 1km 趋势面
        lst_trend_1km = a * ndvi_filled + b * dem_filled + c

        # 2. 计算残差 (在 25km 尺度)
        lst_trend_low = a * ndvi_low + b * dem_low + c
        residual_low = pmw_filled - lst_trend_low

        # 3. 插值残差到 1km (三次卷积平滑)
        residual_1km = cv2.resize(residual_low, (w_1km, h_1km), interpolation=cv2.INTER_CUBIC)

        # 4. 叠加
        lst_final = lst_trend_1km + residual_1km

    except Exception as e:
        print(f"回归失败，使用双线性: {e}")
        lst_final = cv2.resize(pmw_filled, (w_1km, h_1km), interpolation=cv2.INTER_LINEAR)

    return lst_final


# ==============================================================================
# 3. 主程序 (含断点续传)
# ==============================================================================

if __name__ == "__main__":
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)

    print(f"=== 开始降尺度 (掩膜优化版：仅保留原始数据覆盖区) ===")
    print(f"输出目录: {OUT_DIR}")
    print(f"SHAP数据目录: {SHAP_DIR}")

    # 1. 扫描文件
    files_by_dekad = defaultdict(list)
    tif_files = glob.glob(os.path.join(PMW_DIR, "*.tif"))
    for f in tif_files:
        try:
            match = re.search(r'(\d{8})', os.path.basename(f))
            if not match: continue
            date_str = match.group(1)
            date_obj = datetime.strptime(date_str, "%Y%m%d")
            # [可选] 只处理特定月份
            # if date_str[4:6] != '01': continue
            dekad_key = get_ndvi_dekad_date(date_obj)
            files_by_dekad[dekad_key].append((f, date_str))
        except:
            continue
    sorted_dekads = sorted(files_by_dekad.keys())

    # 2. 读取 DEM (作为基础陆地掩膜)
    print("读取 DEM 数据...")
    dem_1km_raw, w_1km, h_1km, geo, proj = read_tif(DEM_FILE)

    # 3. 循环处理
    for dekad_date in tqdm(sorted_dekads, desc="总进度 (旬)"):
        files_in_dekad = files_by_dekad[dekad_date]

        # --- 检查该旬是否全部完成 ---
        all_files_exist = True
        for _, date_str in files_in_dekad:
            expected_out = os.path.join(OUT_DIR, f"FY3C_MWRIX_D_{date_str}_GF_1km_LST.tif")
            if not os.path.exists(expected_out):
                all_files_exist = False
                break

        if all_files_exist:
            tqdm.write(f"  [跳过] 旬 {dekad_date} 已全部完成。")
            continue
        # ----------------------------

        # 读取 NDVI
        ndvi_file = os.path.join(NDVI_DIR, f"{dekad_date}_NDVI.tif")
        if os.path.exists(ndvi_file):
            ndvi_1km_raw, _, _, _, _ = read_tif(ndvi_file)
            if ndvi_1km_raw.shape != dem_1km_raw.shape:
                ndvi_1km_raw = cv2.resize(ndvi_1km_raw, (w_1km, h_1km), interpolation=cv2.INTER_NEAREST)
        else:
            ndvi_1km_raw = None

        # 处理文件
        for lst_file, date_str in tqdm(files_in_dekad, desc=f"  处理 {dekad_date}", leave=False):
            out_name = os.path.join(OUT_DIR, f"FY3C_MWRIX_D_{date_str}_GF_1km_LST.tif")

            if os.path.exists(out_name):
                continue

            try:
                # 读取 PMW
                pmw_arr, _, _, _, _ = read_tif(lst_file)

                # ==========================================================
                # [核心逻辑修改]：构建基于原始 PMW 数据范围的掩膜
                # ==========================================================

                # 1. 提取原始 PMW 数据的有效范围 (非 NaN 区域)
                #    astype(np.float32) 将 True/False 转为 1.0/0.0
                pmw_valid_mask_low = (~np.isnan(pmw_arr)).astype(np.float32)

                # 2. 将有效范围重采样到 1km
                #    必须使用 INTER_NEAREST (最近邻)，保证掩膜边缘清晰，不产生 0.5 这种模糊值
                pmw_valid_mask_1km = cv2.resize(pmw_valid_mask_low, (w_1km, h_1km), interpolation=cv2.INTER_NEAREST)

                # 3. 组合掩膜：只有当 (DEM是陆地) 且 (原始PMW有数据) 时，才保留结果
                #    这样就切除了格陵兰岛那些被 fill_gaps 强行填充出来的马赛克区域
                final_mask = np.where(
                    (~np.isnan(dem_1km_raw)) & (pmw_valid_mask_1km > 0.5),
                    1.0,  # 有效值标记
                    np.nan  # 无效值标记
                )

                # ==========================================================

                # 执行降尺度 (计算过程依然需要 filled 数据来保证回归稳定，并增加保存SHAP数据的传参)
                lst_full = run_downscaling_gwr_filled(
                    pmw_arr, ndvi_1km_raw, dem_1km_raw, w_1km, h_1km,
                    date_str=date_str, shap_out_dir=SHAP_DIR
                )

                # 保存并应用新的组合掩膜 (final_mask)
                save_tif_masked(lst_full, out_name, w_1km, h_1km, geo, proj, mask_arr=final_mask)

            except Exception as e:
                print(f"[错误] {date_str}: {e}")

    print("\n全部完成！")