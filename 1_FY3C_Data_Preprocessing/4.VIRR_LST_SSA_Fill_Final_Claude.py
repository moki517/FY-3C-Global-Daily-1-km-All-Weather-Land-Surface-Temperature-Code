import os.path
import sys
import json
import warnings
import math
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import numpy as np
import pandas as pd
from osgeo import gdal, gdal_array
from tqdm import tqdm
from mssa.mssa import mSSA

# --- Configuration ---
# Filter specific warnings
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed to converge. Check mle_retvals")
# 禁用GDAL异常
gdal.UseExceptions()

# --- Classes ---

class ProgressManager:
    """
    A robust progress manager that uses a JSON file for tracking.
    Handles resumption, statistics, and error logging.
    """
    def __init__(self, progress_file):
        self.progress_file = progress_file
        self.data = {
            'start_time': None,
            'end_time': None,
            'total_chunks': 0,
            'completed_chunks': [],
            'stats': {
                'total_valid_pixels': 0,
                'total_processed_pixels': 0,
                'failed_chunks': []
            }
        }
        self.load()

    def load(self):
        """Loads progress from the JSON file if it exists."""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
                print(f"✓ 已加载进度文件: {self.progress_file}")
                print(f"  已完成块数: {len(self.data['completed_chunks'])}/{self.data.get('total_chunks', 'N/A')}")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"✗ 进度文件损坏或格式不正确 ({e})。将创建新文件。")
                os.rename(self.progress_file, f"{self.progress_file}.bak")

    def save(self):
        """Saves the current progress to the JSON file."""
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"✗ 保存进度文件失败: {e}")

    def initialize(self, total_chunks):
        """Initializes or updates the progress tracker."""
        if self.data['start_time'] is None:
            self.data['start_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.data['total_chunks'] = total_chunks
        self.save()

    def is_completed(self, row_idx, col_idx):
        """Checks if a chunk is already marked as completed."""
        return f"{row_idx}_{col_idx}" in self.data['completed_chunks']

    def mark_completed(self, row_idx, col_idx, valid_count, processed_count):
        """Marks a chunk as completed and updates statistics."""
        chunk_id = f"{row_idx}_{col_idx}"
        if chunk_id not in self.data['completed_chunks']:
            self.data['completed_chunks'].append(chunk_id)
        self.data['stats']['total_valid_pixels'] += valid_count
        self.data['stats']['total_processed_pixels'] += processed_count
        self.save()

    def mark_failed(self, row_idx, col_idx, error_message):
        """Logs a failed chunk."""
        fail_info = {
            'chunk_id': f"{row_idx}_{col_idx}",
            'error': str(error_message),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.data['stats']['failed_chunks'].append(fail_info)
        # Also mark as "completed" to avoid retrying it
        self.mark_completed(row_idx, col_idx, 0, 0)

    def get_remaining_chunks(self, all_chunks):
        """Returns a list of chunks that still need to be processed."""
        return [(r, c) for r, c in all_chunks if not self.is_completed(r, c)]

    def finalize(self):
        """Finalizes the process, saving final time and generating a report."""
        self.data['end_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.save()
        self.generate_report()

    def generate_report(self):
        """Generates a final summary text report."""
        report_file = self.progress_file.replace('.json', '_report.txt')
        summary = self.data
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("="*60 + "\nSSA 处理报告\n" + "="*60 + "\n")
                f.write(f"开始时间: {summary.get('start_time', 'N/A')}\n")
                f.write(f"结束时间: {summary.get('end_time', 'N/A')}\n\n")
                f.write(f"总块数: {summary['total_chunks']}\n")
                f.write(f"成功完成块数: {len(summary['completed_chunks']) - len(summary['stats']['failed_chunks'])}\n")
                f.write(f"失败块数: {len(summary['stats']['failed_chunks'])}\n\n")
                f.write(f"总有效像素时间序列: {summary['stats']['total_valid_pixels']}\n")
                f.write(f"总处理像素时间序列: {summary['stats']['total_processed_pixels']}\n\n")
                if summary['stats']['failed_chunks']:
                    f.write("失败块详情:\n")
                    for fail in summary['stats']['failed_chunks']:
                        f.write(f"  - 块 {fail['chunk_id']}: {fail['error']} @ {fail['timestamp']}\n")
                f.write("="*60 + "\n")
            print(f"✓ 处理报告已生成: {report_file}")
        except Exception as e:
            print(f"✗ 生成报告失败: {e}")


class RWImage:
    """遥感影像读写类"""
    def readimg_allinfo(self, filename):
        data = gdal_array.LoadFile(filename)
        dataset = gdal.Open(filename)
        return data, dataset.GetProjection(), dataset.GetGeoTransform()

    def readimg_bulk(self, filename, x0, y0, xpixels, ypixels):
        data = gdal.Open(filename)
        return data.ReadAsArray(x0, y0, xpixels, ypixels)

    def write_bulk_to_file(self, filename, im_data, x0, y0):
        """向已存在的文件写入块数据"""
        dataset = gdal.Open(filename, gdal.GA_Update)
        if dataset is None:
            raise ValueError(f"无法打开文件用于写入: {filename}")
        dataset.GetRasterBand(1).WriteArray(im_data, x0, y0)
        dataset.FlushCache()

# --- Core Functions ---

def calculate_optimal_chunk_size(rows, cols, days, max_memory_gb=8):
    """计算最优块大小"""
    bytes_per_pixel = 4  # float32
    max_memory_bytes = max_memory_gb * 1024**3 * 0.7  # Use 70% of available memory
    max_pixels_per_chunk = max_memory_bytes / (days * bytes_per_pixel)

    if max_pixels_per_chunk >= rows * cols:
        return rows, cols, 1, 1

    chunk_size = int(math.sqrt(max_pixels_per_chunk))
    chunk_rows = min(rows, max(64, chunk_size))
    chunk_cols = min(cols, max(64, int(max_pixels_per_chunk / chunk_rows)))

    num_row_chunks = math.ceil(rows / chunk_rows)
    num_col_chunks = math.ceil(cols / chunk_cols)

    print(f"内存限制: {max_memory_gb} GB")
    print(f"块大小: {chunk_rows} x {chunk_cols}")
    print(f"块数量: {num_row_chunks} x {num_col_chunks} = {num_row_chunks * num_col_chunks}")
    return int(chunk_rows), int(chunk_cols), num_row_chunks, num_col_chunks


def load_chunk_data_optimized(input_dir, dates_str, start_row, start_col, chunk_rows, chunk_cols, nodata_value=np.nan):
    """优化的块数据加载 - 使用并行读取"""
    rt = RWImage()
    days = len(dates_str)
    chunk_data = np.full((days, chunk_rows, chunk_cols), nodata_value, dtype=np.float32)

    def load_single_date(i, date):
        filename = os.path.join(input_dir, f'{date}_LST.tif')
        if os.path.exists(filename):
            try:
                data = rt.readimg_bulk(filename, start_col, start_row, chunk_cols, chunk_rows)
                return i, data
            except Exception as e:
                print(f"读取块数据失败 {date}: {e}")
        return i, None

    with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() * 2)) as executor:
        results = executor.map(lambda args: load_single_date(*args), enumerate(dates_str))

    for i, data in results:
        if data is not None:
            chunk_data[i, :, :] = data
    return chunk_data


def apply_ssa_vectorized(pixel_data, min_valid_points=5):
    """
    向量化的SSA处理函数 (重建逻辑)。
    - 对整个时间序列进行平滑重建 (所有值都会改变)。
    - 解决了float32溢出警告。
    - 增加了数值稳定性检查。
    """
    days = len(pixel_data)
    valid_mask = ~np.isnan(pixel_data) & (pixel_data > 0)
    valid_count = np.sum(valid_mask)

    if valid_count < min_valid_points:
        return pixel_data

    try:
        data = pd.DataFrame({'y': pixel_data})
        model = mSSA(fill_in_missing=True)
        model.update_model(data[['y']])
        df = model.predict('y', 0, days - 1)
        reconstructed = df['Mean Predictions'].values

        # 1. 检查SSA结果是否有效
        if not np.all(np.isfinite(reconstructed)):
            return pixel_data

        # 2. (关键步骤) 裁剪到float32的安全范围以防止溢出
        f32_info = np.finfo(np.float32)
        reconstructed_clipped = np.clip(reconstructed, f32_info.min, f32_info.max)

        # 3. 安全地转换为float32
        reconstructed_f32 = reconstructed_clipped.astype(np.float32)

        # 保留原始有效数据，只重建缺失值
        # valid_indices = np.where((pixel_values > 0) & ~np.isnan(pixel_values))
        # reconstructed_f32[valid_indices] = reconstructed_f32[valid_indices]

        # 4. (核心改动) 返回完整的重建时间序列
        return reconstructed_f32

    except Exception:
        # 如果SSA过程中发生任何错误，返回原始数据以保证安全
        return pixel_data


def process_pixel_batch(pixel_batch):
    """批处理像素数据"""
    return [apply_ssa_vectorized(pixel) for pixel in pixel_batch]


def apply_ssa_chunk_optimized(chunk_data, nodata_value=np.nan, batch_size=2000,processes_num=8):
    """优化的SSA重建 - 使用批处理和并行计算"""
    days, rows, cols = chunk_data.shape
    chunk_data_filled = np.full_like(chunk_data, nodata_value) # Start with an empty array

    # 找到包含任何有效数据的像素
    max_data = np.nanmax(chunk_data, axis=0)
    valid_mask = ~np.isnan(max_data) & (max_data > 0)
    valid_pixels_coords = np.where(valid_mask)

    # 将无效像素的原始数据复制回去 (通常是NoData)
    invalid_pixels_coords = np.where(~valid_mask)
    chunk_data_filled[:, invalid_pixels_coords[0], invalid_pixels_coords[1]] = chunk_data[:, invalid_pixels_coords[0], invalid_pixels_coords[1]]

    if valid_pixels_coords[0].size == 0:
        return chunk_data_filled, 0, 0

    valid_pixel_timeseries = chunk_data[:, valid_pixels_coords[0], valid_pixels_coords[1]].T
    num_valid_pixels = valid_pixel_timeseries.shape[0]

    num_processes = min(mp.cpu_count(), processes_num)
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Create a dictionary to map futures back to their start index
        futures = {
            executor.submit(process_pixel_batch, valid_pixel_timeseries[i : i + batch_size]): i
            for i in range(0, num_valid_pixels, batch_size)
        }
        # (核心改动) 移除了内部tqdm，保持界面整洁
        for future in futures:
            start_idx = futures[future]
            try:
                batch_results = future.result()
                end_idx = start_idx + len(batch_results)
                # Place results back into the filled array at the correct locations
                rows_to_update = valid_pixels_coords[0][start_idx:end_idx]
                cols_to_update = valid_pixels_coords[1][start_idx:end_idx]
                chunk_data_filled[:, rows_to_update, cols_to_update] = np.array(batch_results).T
            except Exception as e:
                print(f"批处理失败: {e}")

    return chunk_data_filled, num_valid_pixels, num_valid_pixels


def create_output_files(output_dir, dates_str, proj, geotrans, rows, cols):
    """预先创建所有输出文件以支持并行写入"""
    print("预创建输出文件...")
    def create_file(date):
        out_file = os.path.join(output_dir, f'{date}_SSA_LST.tif')
        if not os.path.exists(out_file):
            driver = gdal.GetDriverByName('GTiff')
            dataset = driver.Create(out_file, cols, rows, 1, gdal.GDT_Float32,
                                  options=["COMPRESS=LZW", "TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512"])
            dataset.SetGeoTransform(geotrans)
            dataset.SetProjection(proj)
            dataset.GetRasterBand(1).SetNoDataValue(np.nan)

    with ThreadPoolExecutor(max_workers=min(16, os.cpu_count()*2)) as executor:
        list(tqdm(executor.map(create_file, dates_str), total=len(dates_str), desc="创建文件"))


# --- Main Execution ---

if __name__ == "__main__":
    # 配置参数
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_dir = parent_dir+ r'\Data\VIRR_LST'
    output_dir = parent_dir+ r'\Data\VIRR_LST_SSA'
    begin_date = '20190101'
    end_date = '20191231'
    max_memory_gb = 8
    processes_num = 8

    os.makedirs(output_dir, exist_ok=True)
    progress_file = os.path.join(output_dir, 'processing_progress.json')
    progress_manager = ProgressManager(progress_file)

    try:
        # 1. 获取影像基本信息
        print("="*60 + "\n步骤1: 获取影像基本信息")
        rt = RWImage()
        dates_str = pd.date_range(start=begin_date, end=end_date, freq='D').strftime('%Y%m%d').tolist()

        first_file = next((os.path.join(input_dir, f'{d}_LST.tif') for d in dates_str if os.path.exists(os.path.join(input_dir, f'{d}_LST.tif'))), None)
        if not first_file: raise FileNotFoundError("在指定日期范围内未找到任何有效的影像文件。")

        data, proj, geotrans = rt.readimg_allinfo(first_file)
        rows, cols = data.shape[-2:]
        del data
        print(f"✓ 参考文件: {os.path.basename(first_file)}")
        print(f"  图像尺寸: {rows} x {cols}, 日期数: {len(dates_str)}")

        # 2. 计算处理策略
        print("="*60 + "\n步骤2: 计算处理策略")
        print(f"进程数量: {processes_num}")
        chunk_rows, chunk_cols, num_row_chunks, num_col_chunks = calculate_optimal_chunk_size(rows, cols, len(dates_str), max_memory_gb)
        all_chunks = [(r, c) for r in range(num_row_chunks) for c in range(num_col_chunks)]
        progress_manager.initialize(total_chunks=len(all_chunks))

        # 3. 准备输出文件 (仅在首次运行时)
        if len(progress_manager.data['completed_chunks']) == 0:
            print("="*60 + "\n步骤3: 首次运行，准备输出文件")
            create_output_files(output_dir, dates_str, proj, geotrans, rows, cols)
        else:
            print("="*60 + "\n步骤3: 检测到已有进度，跳过文件预创建")

        # 4. 分块处理
        print("="*60 + "\n步骤4: 开始或恢复分块处理")
        remaining_chunks = progress_manager.get_remaining_chunks(all_chunks)
        if not remaining_chunks:
            print("✓ 所有块均已处理完成！")
        else:
            print(f"总块数: {len(all_chunks)}, 上次处理已完成: {len(all_chunks) - len(remaining_chunks)}, 剩余: {len(remaining_chunks)}")
            with tqdm(total=len(all_chunks), initial=len(all_chunks) - len(remaining_chunks), desc="总进度", unit="块") as pbar:
                for row_idx, col_idx in remaining_chunks:
                    chunk_info = f"块 ({row_idx+1}/{num_row_chunks}, {col_idx+1}/{num_col_chunks})"
                    pbar.set_postfix_str(chunk_info)
                    try:
                        start_row, start_col = row_idx * chunk_rows, col_idx * chunk_cols
                        actual_rows = min(chunk_rows, rows - start_row)
                        actual_cols = min(chunk_cols, cols - start_col)

                        chunk_data = load_chunk_data_optimized(input_dir, dates_str, start_row, start_col, actual_rows, actual_cols)

                        chunk_filled, valid_count, proc_count = apply_ssa_chunk_optimized(chunk_data,processes_num=processes_num)

                        for day_idx, date in enumerate(dates_str):
                            out_file = os.path.join(output_dir, f'{date}_SSA_LST.tif')
                            rt.write_bulk_to_file(out_file, chunk_filled[day_idx, :, :], start_col, start_row)

                        progress_manager.mark_completed(row_idx, col_idx, valid_count, proc_count)
                        del chunk_data, chunk_filled

                    except Exception as e:
                        print(f"✗ 块 {chunk_info} 处理失败: {e}")
                        progress_manager.mark_failed(row_idx, col_idx, e)

                    pbar.update(1)

        # 5. 完成
        print("="*60 + "\n处理完成!")
        progress_manager.finalize()

    except KeyboardInterrupt:
        print("\n\n! 程序被用户中断。当前进度已保存。")
        progress_manager.save()
        sys.exit(1)
    except Exception as e:
        print(f"\n\n! 发生严重错误: {e}")
        import traceback
        traceback.print_exc()
        progress_manager.save()
        sys.exit(1)