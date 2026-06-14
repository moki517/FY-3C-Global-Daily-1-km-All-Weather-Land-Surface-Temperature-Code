# -*- coding: utf-8 -*-
import calendar
import os
import gc
from datetime import datetime, timedelta
import joblib
import numpy as np
from xgboost import XGBRegressor
from osgeo import gdal, gdal_array
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy import stats
from sklearn import metrics
import warnings

# 忽略一些常规警告以保持控制台整洁
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ================= 配置区域 =================
# 数据路径请根据您的实际环境修改
DATA_DIR = r'示例数据集\Data'
NDVI_DIR = os.path.join(DATA_DIR, 'Sub_to_25km', 'VIRR_NDVI_25km')
BASE_DIR = os.path.join(DATA_DIR, 'Base')
PMW_LST_DIR = os.path.join(DATA_DIR, 'Sub_to_25km', 'CDF_Corrected_PMW_LST_25km')
VIRR_LST_DIR = os.path.join(DATA_DIR, 'Sub_to_25km', 'VIRR_LST_25km')
ERA5_DIR = os.path.join(DATA_DIR, 'Sub_to_25km', 'ERA5_25km')
OUT_DIR = os.path.join(DATA_DIR, 'Sub_to_25km', 'PMW_VIRR_Fusion_LST')

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)


# ================= 工具函数 =================

def get_dekad_dates(year):
    """生成一年 36 旬的日期列表"""
    dekad_dates = []
    for month in range(1, 13):
        first_day = datetime(year, month, 1)
        # 第一旬 (1-10日)
        dekad_dates.append([(first_day + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(10)])
        # 第二旬 (11-20日)
        dekad_dates.append([(first_day + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(10, 20)])
        # 第三旬 (21-月底)
        _, num_days = calendar.monthrange(year, month)
        dekad_dates.append([(first_day + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(20, num_days)])
    return dekad_dates


def calculate_confidence_weights(virr, era5, threshold=8.0):
    """
    计算负样本优化的置信度权重
    - 差异 < 3K: 权重为 1.0 (正样本)
    - 差异 3K~8K: 权重线性下降
    - 差异 > 8K: 权重 0.05 (负样本)
    """
    diff = np.abs(virr - era5)
    weights = np.ones_like(virr)

    mask_fade = (diff >= 3.0) & (diff <= threshold)
    weights[mask_fade] = 1.0 - (diff[mask_fade] - 3.0) / (threshold - 3.0) * 0.8

    weights[diff > threshold] = 0.05
    return weights


def BuildOptimizedXGBoost(model_path, X, y, weights, random_state=42):
    """带超参数自主寻优的 XGBoost 训练函数"""
    start_time = datetime.now()

    # 1. 划分数据集（包含权重）
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.3, random_state=random_state)

    print(f"  --> 开始随机超参数搜索，训练集样本量: {len(y_train)}...")

    # 2. 定义基础模型 (开启 GPU 加速，核心保持为 1 防止死锁)
    xgb_base = XGBRegressor(
        device='cuda',
        tree_method='hist',
        random_state=random_state,
        n_jobs=1
    )

    # 3. 定义超参数搜索空间
    param_dist = {
        'n_estimators': stats.randint(200, 550),  # 树的数量
        'max_depth': stats.randint(6, 12),  # 树的最大深度
        'learning_rate': stats.uniform(0.02, 0.1),  # 学习率 [0.02, 0.12]
        'subsample': stats.uniform(0.7, 0.3),  # 样本采样率 [0.7, 1.0]
        'colsample_bytree': stats.uniform(0.7, 0.3),  # 特征采样率 [0.7, 1.0]
        'gamma': stats.uniform(0, 3),  # 最小分裂损失下降
        'min_child_weight': stats.randint(1, 5)  # 最小叶子节点权重和
    }

    # 4. 执行随机搜索 (RandomizedSearchCV)
    # n_iter=20表示尝试20组参数，cv=3表示3折交叉验证。兼顾了精度和耗时。
    search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_dist,
        n_iter=20,
        scoring='neg_root_mean_squared_error',  # 使用 RMSE 作为寻优评价标准
        cv=3,
        n_jobs=2,  # 并行执行2个折叠 (显存允许的话)
        verbose=1,  # 打印搜索进度
        random_state=random_state
    )

    # 【核心】：将 sample_weight 传给 fit 函数，确保交叉验证时权重生效
    search.fit(X_train, y_train, sample_weight=w_train)

    best_model = search.best_estimator_
    print(f"  --> 寻优完成! 本旬最佳超参数: {search.best_params_}")

    # 5. 最终模型评估：仅针对高置信度样本进行真实精度检验
    y_pred = best_model.predict(X_test)
    high_conf = w_test > 0.8
    if np.any(high_conf):
        r2 = metrics.r2_score(y_test[high_conf], y_pred[high_conf])
        rmse = np.sqrt(metrics.mean_squared_error(y_test[high_conf], y_pred[high_conf]))
        mae = metrics.mean_absolute_error(y_test[high_conf], y_pred[high_conf])
        print(f"  --> 高置信度样本评估 -> R2: {r2:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}")

    # 6. 保存模型并记录日志
    joblib.dump(best_model, model_path)
    with open(os.path.join(OUT_DIR, "Training_Log_AutoTune.txt"), "a") as f:
        f.write(f"Model: {os.path.basename(model_path)}\n")
        f.write(f"Best Params: {search.best_params_}\n")
        f.write(f"R2: {r2:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}\n")
        f.write(f"Time Taken: {datetime.now() - start_time}\n")
        f.write("=" * 80 + "\n")

    print(f"本旬总耗时：{datetime.now() - start_time}\n")
    return best_model


# ================= 主程序 =================

if __name__ == '__main__':
    # 1. 读取基础地理数据
    base_file = os.path.join(BASE_DIR, 'FY3_VIRR_NDVI_MAX_2019_25km_ROI.tif')
    roi_mask = gdal_array.LoadFile(base_file)
    Dem = gdal_array.LoadFile(os.path.join(BASE_DIR, 'MERIT_DEM_25km_NAN.tif'))

    # 2. 循环处理 36 旬
    dekad_all_dates = get_dekad_dates(2019)

    for dekad_idx in range(1, 37):
        model_name = f"PMW_VIRR_Fusion_LST_D{dekad_idx:02d}.joblib"
        model_output_path = os.path.join(OUT_DIR, model_name)

        # 断点续传保护
        if os.path.exists(model_output_path) and os.path.getsize(model_output_path) > 1024:
            print(f"第 {dekad_idx} 旬模型已存在且正常，跳过。")
            continue

        dekad_dates = dekad_all_dates[dekad_idx - 1]
        print(f"=== 正在处理第 {dekad_idx} 旬: {dekad_dates[0]} 至 {dekad_dates[-1]} ===")

        # 3. 按旬读取 NDVI
        dekad_last_day = dekad_dates[-1].replace('-', '')
        ndvi_path = os.path.join(NDVI_DIR, f"{dekad_last_day}_NDVI.tif")

        if not os.path.exists(ndvi_path):
            print(f"警告：找不到第 {dekad_idx} 旬的 NDVI 文件 ({ndvi_path})，跳过本旬。")
            continue

        try:
            NDVI = gdal_array.LoadFile(ndvi_path).astype(np.float32)
        except Exception as e:
            print(f"NDVI读取错误: {e}")
            continue

        list_X, list_y, list_w = [], [], []

        # 4. 逐日处理当前旬内的数据
        for date_str in [d.replace('-', '') for d in dekad_dates]:
            virr_path = os.path.join(VIRR_LST_DIR, f"FY3C_VIRR_{date_str}_25km_LST.tif")
            pmw_path = os.path.join(PMW_LST_DIR, f"CDF_Corrected_PMW_LST_{date_str}.tif")
            era5_path = os.path.join(ERA5_DIR, f"skin_temperature_{date_str}_1015LT_3point.tif")

            if not all(os.path.exists(p) for p in [virr_path, pmw_path, era5_path]):
                continue

            VIRR = gdal_array.LoadFile(virr_path).astype(np.float32)
            PMW = gdal_array.LoadFile(pmw_path).astype(np.float32)
            ERA5 = gdal_array.LoadFile(era5_path).astype(np.float32)

            VIRR[VIRR <= 0] = np.nan
            PMW[PMW <= 0] = np.nan

            # 冰雪表面物理屏蔽
            is_ice_snow = (ERA5 < 273.15) & (NDVI < 0.05)

            # 综合数据掩膜（宽泛引入负样本，不做绝对误差 > 8K 的强行剔除）
            valid_mask = (
                    (roi_mask == 1.) &
                    ~np.isnan(VIRR) & ~np.isnan(PMW) & ~np.isnan(NDVI) & ~np.isnan(ERA5) &
                    (VIRR > 220) & (VIRR < 350) &
                    (NDVI > 0) & (Dem > 0) &
                    (~is_ice_snow)
            )

            if np.sum(valid_mask) < 100:
                continue

            y_samples = VIRR[valid_mask]
            era5_samples = ERA5[valid_mask]

            X_samples = np.column_stack([
                PMW[valid_mask],
                NDVI[valid_mask],
                Dem[valid_mask]
            ])

            # 计算包含低权重的负样本优化权重
            weights = calculate_confidence_weights(y_samples, era5_samples, threshold=8.0)

            list_X.append(X_samples)
            list_y.append(y_samples)
            list_w.append(weights)

        # 5. 堆叠本旬数据并训练模型
        if not list_y:
            print(f"警告: 第 {dekad_idx} 旬无有效数据，跳过。\n")
            continue

        final_X = np.vstack(list_X)
        final_y = np.concatenate(list_y)
        final_w = np.concatenate(list_w)

        if len(final_y) < 1000:
            print(f"警告: 第 {dekad_idx} 旬总有效样本量过低 ({len(final_y)})，跳过。\n")
            continue

        del list_X, list_y, list_w, VIRR, PMW, ERA5
        gc.collect()

        # 调用带有超参数自适应学习的函数
        BuildOptimizedXGBoost(model_output_path, final_X, final_y, final_w)

    print("=== 全年所有旬模型处理完成 ===")
