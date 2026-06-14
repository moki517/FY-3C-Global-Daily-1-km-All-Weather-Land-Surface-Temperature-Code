import os
import numpy as np
from osgeo import gdal, gdal_array
from scipy import interpolate
from scipy.stats import percentileofscore, pearsonr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import pandas as pd
import traceback

warnings.filterwarnings('ignore')


class CDFMatcher:
    """CDF matching correction class - Fixed and updated version"""

    def __init__(self, n_quantiles=None, breakpoints=None, min_samples_per_segment=100):
        if n_quantiles is None and breakpoints is None:
            raise ValueError("Either 'n_quantiles' or 'breakpoints' must be provided.")
        if n_quantiles is not None and breakpoints is not None:
            raise ValueError("Provide either 'n_quantiles' or 'breakpoints', not both.")
        
        self.n_quantiles = n_quantiles
        self.breakpoints = breakpoints
        self.min_samples_per_segment = min_samples_per_segment
        self.ref_data = []
        self.tar_data = []
        self.is_fitted = False

    def add_training_data(self, reference_values, target_values):
        ref_flat = np.asarray(reference_values).flatten()
        tar_flat = np.asarray(target_values).flatten()
        valid_mask = np.isfinite(ref_flat) & np.isfinite(tar_flat)
        ref_valid = ref_flat[valid_mask]
        tar_valid = tar_flat[valid_mask]
        if len(ref_valid) > 0:
            self.ref_data.extend(ref_valid)
            self.tar_data.extend(tar_valid)

    def build_cdf_matching_model(self):
        if len(self.ref_data) == 0 or len(self.tar_data) == 0:
            raise ValueError("No training data available")

        ref_array = np.array(self.ref_data)
        tar_array = np.array(self.tar_data)

        if self.breakpoints is not None:
            ref_min, ref_max = np.min(ref_array), np.max(ref_array)
            unique_bps = sorted(list(set([bp for bp in self.breakpoints if ref_min < bp < ref_max])))
            ref_quantiles = np.concatenate(([ref_min], unique_bps, [ref_max]))
            percentiles = [percentileofscore(ref_array, v, kind='strict') for v in ref_quantiles]
            tar_quantiles = np.percentile(tar_array, percentiles)
        else:
            percentiles = np.linspace(0, 100, self.n_quantiles)
            ref_quantiles = np.percentile(ref_array, percentiles)
            tar_quantiles = np.percentile(tar_array, percentiles)

        ref_quantiles = np.sort(ref_quantiles)
        tar_quantiles = np.sort(tar_quantiles)

        try:
            unique_tar, unique_indices = np.unique(tar_quantiles, return_index=True)
            unique_ref = ref_quantiles[unique_indices]
            if len(unique_tar) < 2:
                raise ValueError("Cannot create interpolation function with less than 2 unique points.")
            self.transform_func = interpolate.interp1d(unique_tar, unique_ref, kind='linear', bounds_error=False, fill_value='extrapolate')
            self.is_fitted = True
        except Exception as e:
            print(f"  - Warning: Linear interpolation failed ({e}). Falling back to polynomial fit.")
            try:
                coeffs = np.polyfit(tar_quantiles, ref_quantiles, min(2, len(tar_quantiles) - 1))
                self.transform_func = np.poly1d(coeffs)
                self.is_fitted = True
            except Exception as e2:
                print(f"  - Error: Polynomial fit also failed: {e2}")
                raise

    def apply_correction(self, target_array):
        if not self.is_fitted:
            raise ValueError("Model has not been trained. Please call build_cdf_matching_model() first.")
        
        original_shape = target_array.shape
        target_flat = target_array.flatten()
        corrected_flat = np.full_like(target_flat, np.nan, dtype=np.float64)
        valid_mask = np.isfinite(target_flat)

        if not np.any(valid_mask):
            return target_array.copy()
        
        try:
            corrected_flat[valid_mask] = self.transform_func(target_flat[valid_mask])
        except Exception as e:
            print(f"Error during transformation application: {e}")
            return target_array.copy()

        return corrected_flat.reshape(original_shape)


def save_geotiff(output_path, array, projection, geotransform, no_data_value=-9999):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = array.shape
    dataset = driver.Create(output_path, cols, rows, 1, gdal.GDT_Float32)
    dataset.SetGeoTransform(geotransform)
    dataset.SetProjection(projection)
    band = dataset.GetRasterBand(1)
    band.WriteArray(array.astype(np.float32))
    if no_data_value is not None:
        band.SetNoDataValue(float(no_data_value))
    dataset.FlushCache()


def get_date_list(year):
    start_date = datetime(year, 1, 1)
    is_leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    days_in_year = 366 if is_leap else 365
    dates = [start_date + timedelta(days=i) for i in range(days_in_year)]
    return [date.strftime("%Y%m%d") for date in dates]


def batch_process_cdf_correction(
        year, imputed_lst_dir, reference_lst_dir, output_dir, roi_file,
        n_quantiles=None, breakpoints=None):
    print(f"Starting DAILY batch CDF correction process for the year {year}")
    all_dates = get_date_list(year)
    
    available_files = {}
    for date_str in all_dates:
        imputed_file = os.path.join(imputed_lst_dir, f"Imputed_PMW_LST_{date_str}.tif")
        if os.path.exists(imputed_file):
            available_files[date_str] = imputed_file
    print(f"Found {len(available_files)} imputed LST files to process.")
    if not available_files: return

    try:
        roi_ds = gdal.Open(roi_file)
        roi_mask = gdal_array.LoadFile(roi_file)
        proj = roi_ds.GetProjection()
        geotrans = roi_ds.GetGeoTransform()
    except Exception as e:
        print(f"Fatal Error: Cannot open or read ROI file: {roi_file}. Error: {e}")
        return

    success_count = 0
    for i, (date_str, imputed_file) in enumerate(available_files.items()):
        print(f"\nProcessing file {i+1}/{len(available_files)}: {date_str}")
        try:
            ref_file = os.path.join(reference_lst_dir, f"skin_temperature_{date_str}_1015LT_3point.tif")
            if not os.path.exists(ref_file):
                print(f"  - Skipping: Reference LST file not found for {date_str}")
                continue

            imputed_lst = gdal_array.LoadFile(imputed_file)
            ref_lst = gdal_array.LoadFile(ref_file)
            
            valid_mask = ((roi_mask == 1) & (ref_lst > 220) & (ref_lst < 350) & 
                          (imputed_lst > 220) & (imputed_lst < 350) &
                          np.isfinite(ref_lst) & np.isfinite(imputed_lst))

            if np.sum(valid_mask) < 1000:
                print(f"  - Skipping: Insufficient valid pixels ({np.sum(valid_mask)}) for {date_str}")
                continue

            ref_valid, imputed_valid = ref_lst[valid_mask], imputed_lst[valid_mask]
            
            cdf_matcher = CDFMatcher(n_quantiles=n_quantiles, breakpoints=breakpoints)
            cdf_matcher.add_training_data(ref_valid, imputed_valid)
            
            print(f"  - Found {len(ref_valid)} valid pixels. Building daily CDF model...")
            cdf_matcher.build_cdf_matching_model()
            
            print("  - Applying daily correction...")
            corrected_lst = cdf_matcher.apply_correction(imputed_lst)
            corrected_lst[roi_mask != 1] = -9999

            output_file = os.path.join(output_dir, f"CDF_Corrected_PMW_LST_{date_str}.tif")
            save_geotiff(output_file, corrected_lst, proj, geotrans, no_data_value=-9999)
            
            print(f"  - Success! Saved corrected file to {output_file}")
            success_count += 1
        except Exception as e:
            print(f"  - ERROR processing {date_str}: {e}")
            traceback.print_exc()
            continue
    print(f"\nCDF correction complete! Successfully processed {success_count}/{len(available_files)} files.")

# --- NEW FUNCTION: Validate correction results ---
def validate_correction_results(year, corrected_dir, reference_dir, roi_file, validation_output_dir):
    """
    Compares corrected LST against reference LST and calculates validation statistics.
    """
    print("\n" + "="*50)
    print(f"Starting validation process for the year {year}...")

    os.makedirs(validation_output_dir, exist_ok=True)

    try:
        roi_mask = gdal_array.LoadFile(roi_file)
    except Exception as e:
        print(f"Fatal Error: Cannot read ROI file: {roi_file}. Validation aborted. Error: {e}")
        return

    validation_stats = []
    
    # Find all corrected files
    corrected_files = [f for f in os.listdir(corrected_dir) if f.startswith('CDF_Corrected_') and f.endswith('.tif')]
    
    if not corrected_files:
        print("No corrected files found to validate.")
        return

    print(f"Found {len(corrected_files)} corrected files to validate.")

    for i, fname in enumerate(sorted(corrected_files)):
        date_str = fname.replace('CDF_Corrected_PMW_LST_', '').replace('.tif', '')
        
        # Paths for current day
        corrected_file_path = os.path.join(corrected_dir, fname)
        reference_file_path = os.path.join(reference_dir, f"skin_temperature_{date_str}_1015LT_3point.tif")
        
        if not os.path.exists(reference_file_path):
            continue

        # Load data
        corrected_lst = gdal_array.LoadFile(corrected_file_path)
        reference_lst = gdal_array.LoadFile(reference_file_path)

        # Create a strict validation mask
        # Pixels must be valid in ROI, reference, AND corrected data
        validation_mask = (
            (roi_mask == 1) &
            (reference_lst > 220) & (reference_lst < 350) &
            (corrected_lst > -9999) &  # Use NoDataValue for check
            np.isfinite(reference_lst) & np.isfinite(corrected_lst)
        )
        
        num_valid_pixels = np.sum(validation_mask)
        
        if num_valid_pixels < 100: # Need enough points for meaningful stats
            continue
        
        # Extract valid data
        y_true = reference_lst[validation_mask]
        y_pred = corrected_lst[validation_mask]
        
        # Calculate statistics
        bias = np.mean(y_pred - y_true)
        rmse = np.sqrt(np.mean((y_pred - y_true)**2))
        
        # Calculate R-squared (coefficient of determination)
        # Using pearsonr is more robust than simple np.corrcoef
        r_val, _ = pearsonr(y_pred, y_true)
        r_squared = r_val**2
        
        validation_stats.append({
            'date': pd.to_datetime(date_str),
            'bias': bias,
            'rmse': rmse,
            'r_squared': r_squared,
            'valid_pixels': num_valid_pixels
        })
        
        if (i+1) % 50 == 0:
            print(f"  Validated {i+1}/{len(corrected_files)} files...")

    if not validation_stats:
        print("No valid data pairs found. Could not generate validation report.")
        return

    # Create and save DataFrame
    stats_df = pd.DataFrame(validation_stats).set_index('date')
    csv_path = os.path.join(validation_output_dir, f'Validation_Report_{year}.csv')
    stats_df.to_csv(csv_path)
    print(f"\nValidation statistics saved to: {csv_path}")

    # Plot results
    plot_validation_timeseries(stats_df, year, validation_output_dir)

# --- NEW FUNCTION: Plot validation time series ---
def plot_validation_timeseries(stats_df, year, output_dir):
    """
    Plots the time series of validation metrics.
    """
    if stats_df.empty:
        return
        
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig.suptitle(f'Daily Validation Metrics for CDF Corrected LST - {year}', fontsize=16, fontweight='bold')

    # Bias Plot
    axes[0].plot(stats_df.index, stats_df['bias'], 'o-', markersize=3, color='c', label='Daily Bias')
    mean_bias = stats_df['bias'].mean()
    axes[0].axhline(mean_bias, color='r', linestyle='--', label=f'Mean Bias: {mean_bias:.2f} K')
    axes[0].axhline(0, color='k', linestyle=':', alpha=0.5)
    axes[0].set_ylabel('Bias (K)')
    axes[0].set_title('Daily Bias (Corrected - Reference)')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend()

    # RMSE Plot
    axes[1].plot(stats_df.index, stats_df['rmse'], 'o-', markersize=3, color='m', label='Daily RMSE')
    mean_rmse = stats_df['rmse'].mean()
    axes[1].axhline(mean_rmse, color='r', linestyle='--', label=f'Mean RMSE: {mean_rmse:.2f} K')
    axes[1].set_ylabel('RMSE (K)')
    axes[1].set_title('Daily Root Mean Square Error')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend()

    # R-squared Plot
    axes[2].plot(stats_df.index, stats_df['r_squared'], 'o-', markersize=3, color='g', label='Daily R²')
    mean_r2 = stats_df['r_squared'].mean()
    axes[2].axhline(mean_r2, color='r', linestyle='--', label=f'Mean R²: {mean_r2:.2f}')
    axes[2].set_ylabel('R²')
    axes[2].set_title('Daily R-squared')
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(True, linestyle='--', alpha=0.5)
    axes[2].legend()

    plt.xlabel('Date')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plot_path = os.path.join(output_dir, f'Validation_TimeSeries_{year}.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Validation time series plot saved to: {plot_path}")

def plot_single_day_cdf_analysis(date_str, imputed_lst_dir, reference_lst_dir, roi_file, output_dir, corrected_lst_dir=None):
    # This function remains unchanged.
    print(f"Starting to plot CDF analysis for {date_str}...")
    # ... (code is identical to previous version) ...

def plot_example_cdf_analysis():
    # This function remains unchanged.
    date_str = "20190701"
    # ... (code is identical to previous version) ...


if __name__ == '__main__':
    YEAR = 2019
    
    try: script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError: script_dir = os.getcwd() # Fallback for interactive environments
        
    project_dir = os.path.dirname(os.path.dirname(script_dir))
    data_root_dir = os.path.join(project_dir, 'Data', 'Sub_to_25km')
    
    # --- Define all paths ---
    imputed_lst_dir = os.path.join(data_root_dir, 'Imputed_PMW_LST_25km')
    reference_lst_dir = os.path.join(data_root_dir, 'ERA5_25km')
    roi_file = os.path.join(project_dir, 'Data', 'Base', 'FY3_VIRR_NDVI_MAX_2019_25km_ROI.tif')
    corrected_output_dir = os.path.join(data_root_dir, 'CDF_Corrected_PMW_LST_25km')
    
    # Define a separate directory for validation reports and plots
    validation_output_dir = os.path.join(data_root_dir, 'Validation_Reports')

    # --- Step 1: Run the daily CDF correction ---
    try:
        batch_process_cdf_correction(
            year=YEAR,
            imputed_lst_dir=imputed_lst_dir,
            reference_lst_dir=reference_lst_dir,
            output_dir=corrected_output_dir,
            roi_file=roi_file,
            breakpoints=[260, 310, 330]
        )
    except Exception as e:
        print(f"Batch processing program execution failed: {e}")
        traceback.print_exc()

    # --- Step 2: Run the validation on the results ---
    # This will automatically run after the correction step is complete.
    try:
        validate_correction_results(
            year=YEAR,
            corrected_dir=corrected_output_dir,
            reference_dir=reference_lst_dir,
            roi_file=roi_file,
            validation_output_dir=validation_output_dir
        )
    except Exception as e:
        print(f"Validation program execution failed: {e}")
        traceback.print_exc()

    # --- Step 3: Plot a single-day example analysis chart ---
    print("\n" + "=" * 50)
    print("Plotting example CDF analysis chart for a single day...")
    plot_example_cdf_analysis()