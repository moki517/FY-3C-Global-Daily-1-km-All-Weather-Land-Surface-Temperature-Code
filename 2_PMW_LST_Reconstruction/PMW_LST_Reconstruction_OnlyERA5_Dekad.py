import calendar
import os
from datetime import datetime, timedelta
import joblib
import numpy as np
from xgboost import XGBRegressor
from osgeo import gdal, gdal_array
from scipy import stats
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn import metrics


# --- 功能函数 (与您参考代码中的类似) ---

def get_dekad_dates(year):
    """
    根据输入的年份，生成该年 36 个旬对应的具体日期列表。
    """
    dekad_dates = []
    for month in range(1, 13):
        first_day = datetime(year, month, 1)
        # 第一个旬 (1-10日)
        dekad_1 = [(first_day + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(10)]
        dekad_dates.append(dekad_1)
        # 第二个旬 (11-20日)
        start_date_2 = first_day + timedelta(days=10)
        dekad_2 = [(start_date_2 + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(10)]
        dekad_dates.append(dekad_2)
        # 第三个旬 (21-月底)
        start_date_3 = first_day + timedelta(days=20)
        _, num_days = calendar.monthrange(year, month)
        dekad_3 = [(start_date_3 + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(num_days - 20)]
        dekad_dates.append(dekad_3)
    return dekad_dates


def calculate_metrics(y_true, y_pred):
    """计算一系列标准的回归评估指标。"""
    return (
        metrics.r2_score(y_true, y_pred),
        metrics.explained_variance_score(y_true, y_pred),
        metrics.max_error(y_true, y_pred),
        metrics.mean_absolute_error(y_true, y_pred),
        metrics.mean_squared_error(y_true, y_pred),
        metrics.median_absolute_error(y_true, y_pred),
    )


def BuildXGBoostModel(model_path, y, X, test_size, random_state):
    """训练XGBoost回归模型并评估其性能。"""
    start_time = datetime.now()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    print(f"保存模型至: {model_path}, 测试集比例: {test_size}, 总训练样本数: {X_train.shape[0]}")

    # 初始化 XGBoost 模型，并设置使用 GPU
    xgb = XGBRegressor(tree_method='gpu_hist', random_state=random_state)

    # 使用随机搜索进行超参数调优
    param_random = {
        'n_estimators': stats.randint(100, 500),  # 树的数量
        'max_depth': stats.randint(5, 12),  # 树的最大深度
        'learning_rate': stats.uniform(0.01, 0.1),  # 学习率
        'subsample': stats.uniform(loc=0.8, scale=0.2),  # 抽样比例 (0.8 to 1.0)
        'colsample_bytree': stats.uniform(loc=0.8, scale=0.2),  # 列抽样比例 (0.8 to 1.0)
        'gamma': stats.uniform(0, 5),  # 伽马正则化
        'min_child_weight': stats.randint(1, 5)  # 最小子权重
    }

    # n_iter控制搜索次数，cv是交叉验证折数
    search = RandomizedSearchCV(
        xgb, param_distributions=param_random, n_iter=25,
        scoring='r2', n_jobs=-1, cv=5, verbose=1, random_state=random_state)
    search.fit(X_train, y_train)

    print(f"交叉验证完成 [最佳CV R2 score = {search.best_score_:.3f}]")
    print(f"最佳超参数: {search.best_params_}")

    # 获取最佳模型并评估其在测试集上的性能
    best_model = search.best_estimator_
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)
    y_pred = best_model.predict(X_test)
    regr_metrics = calculate_metrics(y_test, y_pred)

    # 将模型性能和参数记录到文本文件
    stats_file = os.path.join(os.path.dirname(model_path), "Model_Stats_Log.txt")
    with open(stats_file, "a", encoding="utf-8") as f:
        f.write(f"模型: {os.path.basename(model_path)}\n")
        f.write(f"最佳超参数: {search.best_params_}\n")
        f.write(f"测试集 R2 Score = {regr_metrics[0]:.3f}\n")
        f.write(f"测试集 Explained Variance Score = {regr_metrics[1]:.3f}\n")
        f.write(f"测试集 Max Error = {regr_metrics[2]:.2f}\n")
        f.write(f"测试集 Mean Absolute Error = {regr_metrics[3]:.2f}\n")
        f.write(f"测试集 Mean Squared Error = {regr_metrics[4]:.2f}\n")
        f.write(f"测试集 Median Absolute Error = {regr_metrics[5]:.2f}\n")
        f.write(f"训练时间: {datetime.now()}\n")
        f.write("=" * 80 + "\n")

    consume_time = datetime.now() - start_time
    print(f"本旬模型训练总耗时: {consume_time}\n")

    return best_model


def save_geotiff(output_path, array, projection, geotransform, no_data_value=0.0):
    """将Numpy数组保存为GeoTIFF文件。"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = array.shape
    dataset = driver.Create(output_path, cols, rows, 1, gdal.GDT_Float32)
    dataset.SetGeoTransform(geotransform)
    dataset.SetProjection(projection)
    band = dataset.GetRasterBand(1)
    band.WriteArray(array)
    if no_data_value is not None:
        band.SetNoDataValue(float(no_data_value))
    dataset.FlushCache()
    dataset = None


if __name__ == '__main__':
    # --- 1. 路径和参数设置 ---
    YEAR = 2019  # 设置处理年份

    # --- 请根据您的实际情况修改这里的路径 ---
    # 假设此脚本位于 'Your_Project/Scripts/' 目录中
    # Data 目录结构为 'Your_Project/Data/'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(script_dir))
    data_root_dir = os.path.join(project_dir, 'Data', 'Sub_to_25km')

    # 输入数据目录 (移除了NDVI目录)
    PMW_LST_dir = os.path.join(data_root_dir, 'PMW_LST_RAW_25km')
    ERA5_dir = os.path.join(data_root_dir, 'ERA5_25km')
    base_dir = os.path.join(project_dir, 'Data', 'Base')

    # 输出目录
    model_output_dir = os.path.join(data_root_dir, 'Impute_PMW_LST_Models_ERA5_Only')
    imputed_LST_dir = os.path.join(data_root_dir, 'Imputed_PMW_LST_25km_ERA5_Only')

    # ERA5变量列表 (文件名中的变量部分)
    ERA5_VARIABLES = [
        "skin_temperature", "2m_temperature", "surface_net_solar_radiation",
        "surface_net_thermal_radiation", "volumetric_soil_water_layer_1", "2m_dewpoint_temperature"
    ]


    # --- 2. 读取基础数据 ---
    print("正在读取基础数据(ROI)...")
    base_file = os.path.join(base_dir, 'FY3_VIRR_NDVI_MAX_2019_25km_ROI.tif')
    base_ds = gdal.Open(base_file)
    proj = base_ds.GetProjection()
    geotrans = base_ds.GetGeoTransform()
    roi_mask = gdal_array.LoadFile(base_file)
    base_ds = None

    # --- 3. 按旬循环处理 ---
    for dekad_num in range(1, 37):
        dekad_dates = get_dekad_dates(YEAR)
        dekad_date_list = dekad_dates[dekad_num - 1]
        print(f"\n{'=' * 25} 开始处理 {YEAR}年 第 {dekad_num} 旬 {'=' * 25}")
        print(f"日期范围: {dekad_date_list[0]} to {dekad_date_list[-1]}")

        # --- 3.1 训练阶段: 为当前旬收集所有有效像元 ---
        print("\n--- 阶段 1: 收集训练数据 (仅使用ERA5变量) ---")

        # 收集本旬所有天的有效数据 (不再收集NDVI数据)
        training_data_collectors = {'PMW_LST': []}
        for key in ERA5_VARIABLES:
            training_data_collectors[key] = []

        for date_str in dekad_date_list:
            doy_str = date_str.replace('-', '')
            try:
                # 加载当日LST
                pmw_lst_file = os.path.join(PMW_LST_dir, f"FY3C_PMW_{doy_str}_25km_LST.tif")
                pmw_lst = gdal_array.LoadFile(pmw_lst_file)
                pmw_lst[(pmw_lst <= 220) | (pmw_lst > 350)] = np.nan  # 过滤异常LST值

                # 加载当日所有ERA5变量
                era5_arrays = {}
                for var in ERA5_VARIABLES:
                    era5_file = os.path.join(ERA5_dir, f"{var}_{doy_str}_1015LT_3point.tif")
                    era5_data = gdal_array.LoadFile(era5_file)
                    # 假设-9999为无效值, 请根据你的数据修改
                    # era5_data[era5_data < -999] = np.nan
                    era5_arrays[var] = era5_data

                # 构建掩膜，找到LST有效且在ROI内的位置用于训练 (移除了NDVI条件)
                valid_mask = ~np.isnan(pmw_lst) & (roi_mask == 1.)
                # for var in ERA5_VARIABLES:
                #     valid_mask &= ~np.isnan(era5_arrays[var])

                valid_index = np.where(valid_mask)
                if valid_index[0].size == 0:
                    continue

                # 提取有效像元数据 (不再提取NDVI数据)
                training_data_collectors['PMW_LST'].append(pmw_lst[valid_index])
                for var in ERA5_VARIABLES:
                    training_data_collectors[var].append(era5_arrays[var][valid_index])
            except Exception as e:
                print(f"提示: 处理 {date_str} 数据时跳过: {e}")
                continue

        # 检查是否收集到足够的训练数据
        if not training_data_collectors['PMW_LST']:
            print(f"警告: 第 {dekad_num} 旬没有收集到任何有效像元用于训练，无法建立模型。")
            continue

        # 将列表数据合并成Numpy数组 (仅使用ERA5变量)
        y_train_dekad = np.concatenate(training_data_collectors['PMW_LST'])

        predictor_list = ERA5_VARIABLES  # 移除了NDVI
        X_train_list = [np.concatenate(training_data_collectors[p])[:, np.newaxis] for p in predictor_list]
        X_train_dekad = np.hstack(X_train_list)

        print(f"本旬总可用训练样本数: {X_train_dekad.shape[0]}")
        if X_train_dekad.shape[0] < 500:  # 如果样本太少，模型可能不稳定
            print(f"警告: 第 {dekad_num} 旬训练数据过少 ({X_train_dekad.shape[0]}个), 可能导致模型不稳定。")
            # continue # 可以选择跳过

        # --- 3.2 训练XGBoost模型 ---
        print("\n--- 阶段 2: 训练XGBoost模型 (仅基于ERA5变量) ---")
        model_path = os.path.join(model_output_dir, f"PMW_LST_Impute_Model_ERA5Only_Dekad_{dekad_num}.joblib")
        if os.path.exists(model_path):
            print(f"警告: 模型文件 '{model_path}' 已存在，跳过训练，直接插补。")
            best_model = joblib.load(model_path)
        else:
            best_model = BuildXGBoostModel(
                model_path=model_path,
                y=y_train_dekad,
                X=X_train_dekad,
                test_size=0.3,
                random_state=42
            )

        # --- 3.3 插补阶段: 应用模型填充缺失值 ---
        print("\n--- 阶段 3: 插补并保存每日数据 (仅基于ERA5变量) ---")
        for date_str in dekad_date_list:
            doy_str = date_str.replace('-', '')
            output_filename = f"Imputed_PMW_LST_{doy_str}_ERA5Only.tif"
            output_path = os.path.join(imputed_LST_dir, output_filename)

            try:
                # 重新加载当日原始LST数据
                pmw_lst_file = os.path.join(PMW_LST_dir, f"FY3C_PMW_{doy_str}_25km_LST.tif")
                pmw_lst_orig = gdal_array.LoadFile(pmw_lst_file).astype(float)

                imputed_lst = np.copy(pmw_lst_orig)  # 创建输出数组
                all_imputed_lst = np.copy(pmw_lst_orig)  # 创建输出数组

                # 加载所有ERA5预测变量 (不再加载NDVI)
                era5_arrays = {}
                for var in ERA5_VARIABLES:
                    era5_file = os.path.join(ERA5_dir, f"{var}_{doy_str}_1015LT_3point.tif")
                    era5_data = gdal_array.LoadFile(era5_file)
                    # era5_data[era5_data < -999] = np.nan
                    era5_arrays[var] = era5_data

                # 找到需要插补的像元: LST缺失，但ERA5变量都存在 (不再考虑NDVI)
                impute_mask = np.isnan(pmw_lst_orig) & (roi_mask == 1.)
                # for var in ERA5_VARIABLES:
                #     impute_mask &= ~np.isnan(era5_arrays[var])

                impute_index = np.where(impute_mask)

                if impute_index[0].size > 0:
                    # 准备用于预测的数据 (X_impute) - 仅使用ERA5变量
                    X_impute_list = []
                    for var in ERA5_VARIABLES:
                        X_impute_list.append(era5_arrays[var][impute_index][:, np.newaxis])
                    X_impute = np.hstack(X_impute_list)

                    # 执行预测
                    predicted_values = best_model.predict(X_impute)

                    # 将预测值填充到输出数组中
                    imputed_lst[impute_index] = predicted_values

                    # 为所有ROI内的像元进行预测 (全图预测)
                    valid_roi_index = np.where(roi_mask == 1.)

                    X_all_impute_list = []
                    for var in ERA5_VARIABLES:
                        X_all_impute_list.append(era5_arrays[var][valid_roi_index][:, np.newaxis])
                    X_all_impute = np.hstack(X_all_impute_list)

                    # 执行全图预测
                    predicted_all_values = best_model.predict(X_all_impute)
                    all_imputed_lst[valid_roi_index] = predicted_all_values

                    print(f"成功为 {date_str} 插补了 {impute_index[0].size} 个像元。")
                else:
                    print(f"{date_str} 没有需要插补的像元。")

                # 保存插补后的GeoTIFF文件 (无论是否插补都保存，以保证文件完整性)
                save_geotiff(output_path, imputed_lst, proj, geotrans, no_data_value=0.0)

                save_geotiff(output_path.replace('.tif', '_Predicted.tif'), all_imputed_lst, proj, geotrans,
                             no_data_value=0.0)

            except Exception as e:
                print(f"严重警告: 插补或保存 {date_str} 数据时出错: {e}")
                continue

    print(f"\n{'=' * 25} {YEAR}年全部处理完成 (仅使用ERA5变量) {'=' * 25}")