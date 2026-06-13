# -*- coding: utf-8 -*-
# PMW_VIRR_LST_Fusion_NoERA5.py
# 最终融合版：不依赖 ERA5，仅使用 XGBoost 预测 + 残差重建 + PMW云检测

import os
import joblib
import numpy as np
from osgeo import gdal, gdal_array
import sys
from datetime import datetime
import traceback


# ==============================================================================
# Helper Functions (辅助函数)
# ==============================================================================

def _get_dekad_dates_and_end_date(year, dekad_num):
    """获取旬日期列表"""
    try:
        from PMW_VIRR_LST_Fusion1 import get_dekad_dates
        all_dekads = get_dekad_dates(year)
        dekad_dates_list = all_dekads[dekad_num - 1]
        dekad_end_date_str = dekad_dates_list[-1].replace('-', '')
        return dekad_dates_list, dekad_end_date_str
    except ImportError:
        print("Error: Could not import 'get_dekad_dates'. Please ensure 'PMW_VIRR_LST_Fusion1.py' is in the directory.")
        sys.exit(1)


def _warp_raster(src_ds, ref_ds, resampling_method_str, nodata_val):
    """栅格重采样"""
    resampling_map = {"average": gdal.GRA_Average, "cubicspline": gdal.GRA_CubicSpline,
                      "near": gdal.GRA_NearestNeighbour}
    ref_proj, ref_geotransform = ref_ds.GetProjection(), ref_ds.GetGeoTransform()
    x_res, y_res = ref_geotransform[1], abs(ref_geotransform[5])
    min_x, max_y = ref_geotransform[0], ref_geotransform[3]
    max_x = min_x + (ref_geotransform[1] * ref_ds.RasterXSize)
    min_y = max_y + (ref_geotransform[5] * ref_ds.RasterYSize)
    src_nodata = src_ds.GetRasterBand(1).GetNoDataValue()
    if src_nodata is None: src_nodata = np.nan
    warp_options = gdal.WarpOptions(format="VRT", outputBounds=(min_x, min_y, max_x, max_y), xRes=x_res, yRes=y_res,
                                    dstSRS=ref_proj, srcNodata=src_nodata, dstNodata=nodata_val,
                                    resampleAlg=resampling_map[resampling_method_str])
    return gdal.Warp('', src_ds, options=warp_options)


def _array_to_gdal_dataset(array, ref_ds):
    """将numpy数组转换为内存中的GDAL Dataset"""
    rows, cols = array.shape
    driver = gdal.GetDriverByName('MEM')
    ds = driver.Create('', cols, rows, 1, gdal.GDT_Float32)
    ds.SetGeoTransform(ref_ds.GetGeoTransform())
    ds.SetProjection(ref_ds.GetProjection())
    band = ds.GetRasterBand(1)
    band.WriteArray(array)
    band.SetNoDataValue(np.nan)
    band.FlushCache()
    return ds


def save_array_as_geotiff(output_path, array, ref_ds, nodata_val=-9999):
    """保存为GeoTIFF"""
    print(f"    Saving output to {os.path.basename(output_path)}...")
    driver = gdal.GetDriverByName("GTiff")
    rows, cols = array.shape
    out_ds = driver.Create(output_path, cols, rows, 1, gdal.GDT_Float32, options=["COMPRESS=LZW", "BIGTIFF=IF_NEEDED"])
    out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
    out_ds.SetProjection(ref_ds.GetProjection())
    out_band = out_ds.GetRasterBand(1)
    array_copy = array.copy()
    array_copy[np.isnan(array_copy)] = nodata_val
    out_band.WriteArray(array_copy)
    out_band.SetNoDataValue(nodata_val)
    out_band.FlushCache()
    out_ds = None
    print("    Save complete.")


def is_valid_file(filepath):
    """断点续传检查"""
    if not os.path.exists(filepath): return False
    if os.path.getsize(filepath) < 1024: return False
    try:
        ds = gdal.Open(filepath)
        if ds is None: return False
        ds = None
    except:
        return False
    return True


# ==============================================================================
# Core Logic (核心逻辑)
# ==============================================================================

def apply_cloud_consistency_check(virr_arr, pmw_arr, dem_arr, ndvi_arr,
                                  base_threshold=-8.0, glacier_threshold=-18.0):
    """
    云检测：VIRR < PMW - Threshold
    注意：这里的阈值建议设置为 -10.0 或 -12.0，与训练时的清洗标准接近。
    """
    diff = virr_arr - pmw_arr
    threshold_grid = np.full(diff.shape, base_threshold, dtype=np.float32)

    is_glacier_suspect = np.zeros(diff.shape, dtype=bool)
    if dem_arr is not None: is_glacier_suspect |= (dem_arr > 4000)
    if ndvi_arr is not None: is_glacier_suspect |= (ndvi_arr < 0.05)

    threshold_grid[is_glacier_suspect] = glacier_threshold

    # 如果 VIRR 比 PMW 低太多，认为是云
    cloud_mask = ((~np.isnan(virr_arr)) & (~np.isnan(pmw_arr)) & (diff < threshold_grid))

    virr_cleaned = virr_arr.copy()
    virr_cleaned[cloud_mask] = np.nan
    return virr_cleaned


def calculate_coarse_scale_residuals(model, virr_25km_ds, pmw_25km_ds, ndvi_25km_ds, dem_25km_ds):
    """
    计算残差 (不使用 ERA5)
    """
    virr_arr = virr_25km_ds.ReadAsArray().astype(np.float32)
    pmw_arr = pmw_25km_ds.ReadAsArray().astype(np.float32)
    ndvi_arr = ndvi_25km_ds.ReadAsArray().astype(np.float32)
    dem_arr = dem_25km_ds.ReadAsArray().astype(np.float32)

    # Nodata 处理
    for arr, ds in [(virr_arr, virr_25km_ds), (pmw_arr, pmw_25km_ds),
                    (ndvi_arr, ndvi_25km_ds), (dem_arr, dem_25km_ds)]:
        nodata = ds.GetRasterBand(1).GetNoDataValue()
        if nodata is not None: arr[arr == nodata] = np.nan

    # 1. 前置云检测 (PMW vs VIRR)
    # 这一步非常重要：如果 VIRR 是云，将其设为 NaN，这样计算出的 Residual 就是 NaN (或0)，
    # 从而迫使最终结果使用 XGB 的预测值，而不是错误的 VIRR 值。
    virr_arr = apply_cloud_consistency_check(virr_arr, pmw_arr, dem_arr, ndvi_arr)

    # 2. 预测背景场 (Prediction)
    valid_mask = ~np.isnan(pmw_arr) & ~np.isnan(ndvi_arr) & ~np.isnan(dem_arr)
    X_coarse = np.vstack((pmw_arr[valid_mask], ndvi_arr[valid_mask], dem_arr[valid_mask])).T

    if X_coarse.shape[0] == 0: return None

    predicted_lst_flat = model.predict(X_coarse)
    predicted_lst_25km = np.full(pmw_arr.shape, np.nan, dtype=np.float32)
    predicted_lst_25km[valid_mask] = predicted_lst_flat

    # 3. 计算残差 (Residuals)
    # Residual = Observation - Prediction
    raw_residuals = virr_arr - predicted_lst_25km

    # 简单的残差过滤 (防止极端值)
    # 既然已经做了 PMW 云检测，这里只需要过滤极端的数学异常
    residuals_25km = np.full(pmw_arr.shape, 0.0, dtype=np.float32)

    valid_residual_mask = (
            (~np.isnan(raw_residuals)) &
            (raw_residuals > -20.0) &  # 略微放宽，因为已经有前置云检测
            (raw_residuals < 20.0)
    )
    residuals_25km[valid_residual_mask] = raw_residuals[valid_residual_mask]

    return residuals_25km


def predict_fine_scale_lst(model, pmw_1km_ds, ndvi_1km_ds, dem_1km_ds, chunk_size):
    """1km 预测 (纯XGB)"""
    rows, cols = pmw_1km_ds.RasterYSize, pmw_1km_ds.RasterXSize
    pmw_arr = pmw_1km_ds.ReadAsArray().astype(np.float32)
    ndvi_arr = ndvi_1km_ds.ReadAsArray().astype(np.float32)
    dem_arr = dem_1km_ds.ReadAsArray().astype(np.float32)

    for arr, ds in [(pmw_arr, pmw_1km_ds), (ndvi_arr, ndvi_1km_ds), (dem_arr, dem_1km_ds)]:
        nodata = ds.GetRasterBand(1).GetNoDataValue()
        if nodata is not None: arr[arr == nodata] = np.nan

    valid_mask = ~np.isnan(pmw_arr) & ~np.isnan(ndvi_arr) & ~np.isnan(dem_arr)
    X_fine_all = np.vstack((pmw_arr[valid_mask], ndvi_arr[valid_mask], dem_arr[valid_mask])).T

    del pmw_arr, ndvi_arr, dem_arr

    num_valid_pixels = X_fine_all.shape[0]
    if num_valid_pixels == 0: return np.full((rows, cols), np.nan, dtype=np.float32)

    predicted_lst_flat = np.zeros(num_valid_pixels, dtype=np.float32)
    for i in range(0, num_valid_pixels, chunk_size):
        chunk_end = min(i + chunk_size, num_valid_pixels)
        chunk_X = X_fine_all[i:chunk_end, :]
        predicted_lst_flat[i:chunk_end] = model.predict(chunk_X)

    del X_fine_all

    predicted_lst_1km = np.full((rows, cols), np.nan, dtype=np.float32)
    predicted_lst_1km[valid_mask] = predicted_lst_flat
    return predicted_lst_1km


# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == '__main__':
    start_time_total = datetime.now()

    # --- 1. 配置路径 (请确认路径正确) ---
    data_dir = r'F:\Global_FY_LST_2019\FY3C_2019\Data'

    PMW_LST_25km_dir = os.path.join(data_dir, 'Sub_to_25km', 'CDF_Corrected_PMW_LST_25km')
    NDVI_25km_dir = os.path.join(data_dir, 'Sub_to_25km', 'VIRR_NDVI_25km')
    VIRR_LST_25km_dir = os.path.join(data_dir, 'Sub_to_25km', 'VIRR_LST_25km')
    DEM_25km_file = os.path.join(data_dir, 'Base', 'MERIT_DEM_25km_NAN.tif')
    # ERA5 路径已移除

    PMW_LST_1km_dir = os.path.join(data_dir, 'PMW_LST_Downscale_GWR_1')
    virr_1km_dir = os.path.join(data_dir, 'VIRR_LST')
    NDVI_1km_dir = os.path.join(data_dir, 'VIRR_10days_NDVI_GF')
    DEM_1km_file = os.path.join(data_dir, 'Base', 'MERIT_DEM_1D_NAN.tif')

    model_dir = os.path.join(data_dir, 'Sub_to_25km', 'PMW_VIRR_Fusion_LST_4')  # 指向新训练的模型目录
    output_dir = os.path.join(data_dir, 'Global_1km_Fusion_LST_5')
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    YEAR = 2019
    CHUNK_SIZE = 15_000_000

    # --- 2. 循环处理 ---
    for dekad_num in range(1, 37):
        print("-" * 80)
        print(f"Processing Dekad {dekad_num}/{36} for year {YEAR}")

        try:
            model_file = os.path.join(model_dir, f"PMW_VIRR_Fusion_LST_{dekad_num}.joblib")
            if not os.path.exists(model_file):
                print(f"  WARNING: Model file not found, skipping dekad {dekad_num}.")
                continue
            xgb_model = joblib.load(model_file)

            dekad_dates_list, dekad_end_date_str = _get_dekad_dates_and_end_date(YEAR, dekad_num)

            ndvi_25km_file = os.path.join(NDVI_25km_dir, f"{dekad_end_date_str}_NDVI.tif")
            ndvi_1km_file = os.path.join(NDVI_1km_dir, f"{dekad_end_date_str}_NDVI.tif")

            dem_25km_ds = gdal.Open(DEM_25km_file)
            dem_1km_ds = gdal.Open(DEM_1km_file)
            ndvi_25km_ds = gdal.Open(ndvi_25km_file)
            ndvi_1km_ds = gdal.Open(ndvi_1km_file)

            if not all([dem_25km_ds, dem_1km_ds, ndvi_25km_ds, ndvi_1km_ds]): continue

            for date_str in dekad_dates_list:
                yyyymmdd = date_str.replace('-', '')
                print(f"\n  -- Processing Date: {date_str} --")

                output_filename_XGB = f"Global_XGB_Predicted_LST_{yyyymmdd}_1km.tif"
                output_filename_Fusion = f"Global_FY3C_VIRR_PMW_Fusion_LST_{yyyymmdd}_1km.tif"
                output_filepath_XGB = os.path.join(output_dir, output_filename_XGB)
                output_filepath_Fusion = os.path.join(output_dir, output_filename_Fusion)

                if is_valid_file(output_filepath_XGB) and is_valid_file(output_filepath_Fusion):
                    print(f"    [Resume] Skipping date {date_str}.")
                    continue

                daily_start_time = datetime.now()
                try:
                    pmw_25km_file = os.path.join(PMW_LST_25km_dir, f"CDF_Corrected_PMW_LST_{yyyymmdd}.tif")
                    virr_25km_file = os.path.join(VIRR_LST_25km_dir, f"FY3C_VIRR_{yyyymmdd}_25km_LST.tif")
                    pmw_1km_file = os.path.join(PMW_LST_1km_dir, f"FY3C_MWRIX_D_{yyyymmdd}_GF_1km_LST.tif")
                    virr_1km_file = os.path.join(virr_1km_dir, f"{yyyymmdd}_LST.tif")
                    # ERA5 文件检查已移除

                    files_to_check = [pmw_25km_file, virr_25km_file, pmw_1km_file, virr_1km_file]
                    if not all(os.path.exists(f) for f in files_to_check):
                        print(f"    Skipping: Missing files for {yyyymmdd}.")
                        continue

                    pmw_25km_ds = gdal.Open(pmw_25km_file)
                    virr_25km_ds = gdal.Open(virr_25km_file)
                    pmw_1km_ds = gdal.Open(pmw_1km_file)
                    virr_1km_data = gdal_array.LoadFile(virr_1km_file)

                    if not all([pmw_25km_ds, virr_25km_ds, pmw_1km_ds]): continue

                    # Step 1: 25km 计算 (仅计算 Residuals，无 Bias)
                    print(f"    Step 1: Calculating residuals (No ERA5)...")
                    residuals_25km = calculate_coarse_scale_residuals(
                        xgb_model, virr_25km_ds, pmw_25km_ds, ndvi_25km_ds, dem_25km_ds
                    )
                    if residuals_25km is None: continue

                    # Step 2: 降尺度 Residuals
                    residuals_25km_ds = _array_to_gdal_dataset(residuals_25km, virr_25km_ds)
                    residuals_1km_ds = _warp_raster(residuals_25km_ds, pmw_1km_ds, 'cubicspline', 0)
                    residuals_1km = residuals_1km_ds.ReadAsArray()
                    residuals_1km[np.isnan(residuals_1km)] = 0.0

                    # Step 3: 1km 预测
                    print(f"    Step 3: Predicting fine-scale LST...")
                    predicted_LST_1km = predict_fine_scale_lst(xgb_model, pmw_1km_ds, ndvi_1km_ds, dem_1km_ds,
                                                               CHUNK_SIZE)

                    # Step 4: 生成 XGB 融合结果 (仅 Prediction + Residuals)
                    print(f"    Step 4: Applying Residuals...")
                    xgb_LST_1km = np.add(predicted_LST_1km, residuals_1km, where=~np.isnan(predicted_LST_1km))

                    xgb_LST_1km[(xgb_LST_1km > 330) | (xgb_LST_1km < 220)] = np.nan

                    save_array_as_geotiff(output_filepath_XGB, xgb_LST_1km, pmw_1km_ds)

                    # Step 5: 最终融合 (填补 VIRR)
                    print(f"    Step 5: Fusing VIRR and XGB...")
                    pmw_1km_arr = pmw_1km_ds.ReadAsArray().astype(np.float32)
                    dem_1km_arr = dem_1km_ds.ReadAsArray().astype(np.float32)
                    ndvi_1km_arr = ndvi_1km_ds.ReadAsArray().astype(np.float32)

                    virr_1km_clean = apply_cloud_consistency_check(virr_1km_data, pmw_1km_arr, dem_1km_arr,
                                                                   ndvi_1km_arr)

                    mask_nan = np.isnan(virr_1km_clean)
                    virr_1km_clean[mask_nan] = xgb_LST_1km[mask_nan]

                    save_array_as_geotiff(output_filepath_Fusion, virr_1km_clean, pmw_1km_ds)
                    print(f"----Finished date {date_str} in {datetime.now() - daily_start_time}----")

                except Exception as e:
                    print(f"    ERROR processing date {date_str}: {e}")
                    traceback.print_exc()
                    continue

        except Exception as e:
            print(f"  FATAL ERROR processing dekad {dekad_num}: {e}")
            traceback.print_exc()
            continue

    total_time = datetime.now() - start_time_total
    print("=" * 80)
    print(f"All dekads processed. Total time taken: {total_time}")