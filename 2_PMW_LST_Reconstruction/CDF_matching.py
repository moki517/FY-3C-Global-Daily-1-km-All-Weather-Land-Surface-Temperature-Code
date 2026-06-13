import os
import numpy as np
from osgeo import gdal, gdal_array
from scipy import interpolate
from scipy.stats import percentileofscore
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import pandas as pd
import traceback

warnings.filterwarnings('ignore')


class CDFMatcher:
    """CDF matching correction class - Fixed and updated version"""

    def __init__(self, n_quantiles=None, breakpoints=None, min_samples_per_segment=100):
        """
        Initializes the CDF matcher. Provide either n_quantiles or a list of temperature breakpoints.

        Args:
        n_quantiles (int, optional): Number of segments to divide the CDF into. Defaults to None.
        breakpoints (list, optional): A list of temperature values (e.g., [260, 310, 330]) to use as segment nodes. Defaults to None.
        min_samples_per_segment (int): Minimum number of samples per segment.
        """
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
        """Adds training data"""
        ref_flat = np.asarray(reference_values).flatten()
        tar_flat = np.asarray(target_values).flatten()
        valid_mask = np.isfinite(ref_flat) & np.isfinite(tar_flat)
        ref_valid = ref_flat[valid_mask]
        tar_valid = tar_flat[valid_mask]
        if len(ref_valid) > 0:
            self.ref_data.extend(ref_valid)
            self.tar_data.extend(tar_valid)

    def build_cdf_matching_model(self):
        """Builds the CDF matching model"""
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
            self.transform_func = interpolate.interp1d(unique_tar, unique_ref, kind='linear', bounds_error=False,
                                                       fill_value='extrapolate')
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
        """Applies CDF matching correction"""
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
    """Saves a GeoTIFF file"""
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
    band.FlushCache()
    dataset.FlushCache()
    dataset = None


def get_date_list(year):
    """Gets a list of all dates for a given year"""
    start_date = datetime(year, 1, 1)
    is_leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    days_in_year = 366 if is_leap else 365
    dates = [start_date + timedelta(days=i) for i in range(days_in_year)]
    return [date.strftime("%Y%m%d") for date in dates]


# --- MODIFIED FUNCTION: Performs CDF correction on a daily basis ---
def batch_process_cdf_correction(
        year,
        imputed_lst_dir,
        reference_lst_dir,
        output_dir,
        roi_file,
        n_quantiles=None,
        breakpoints=None
):
    """
    Performs batch CDF correction by creating a unique model for each day.
    """
    print(f"Starting DAILY batch CDF correction process for the year {year}")

    all_dates = get_date_list(year)

    # --- Find all available imputed files first ---
    available_files = {}
    for date_str in all_dates:
        imputed_file = os.path.join(imputed_lst_dir, f"Imputed_PMW_LST_{date_str}.tif")
        if os.path.exists(imputed_file):
            available_files[date_str] = imputed_file

    print(f"Found {len(available_files)} imputed LST files to process.")

    if not available_files:
        print("No imputed LST files found, exiting program.")
        return

    # --- Load ROI mask once ---
    try:
        roi_ds = gdal.Open(roi_file)
        roi_mask = gdal_array.LoadFile(roi_file)
        proj = roi_ds.GetProjection()
        geotrans = roi_ds.GetGeoTransform()
        roi_ds = None
    except Exception as e:
        print(f"Fatal Error: Cannot open or read ROI file: {roi_file}. Error: {e}")
        return

    success_count = 0
    # --- Main loop: Process each day individually ---
    for i, (date_str, imputed_file) in enumerate(available_files.items()):

        print(f"\nProcessing file {i + 1}/{len(available_files)}: {date_str}")

        try:
            # --- 1. Load data for the current day ---
            ref_file = os.path.join(reference_lst_dir, f"skin_temperature_{date_str}_1015LT_3point.tif")
            if not os.path.exists(ref_file):
                print(f"  - Skipping: Reference LST file not found for {date_str}")
                continue

            imputed_lst = gdal_array.LoadFile(imputed_file)
            ref_lst = gdal_array.LoadFile(ref_file)

            # --- 2. Find valid overlapping pixels for training ---
            valid_mask = (
                    (roi_mask == 1) &
                    (ref_lst > 220) & (ref_lst < 350) &
                    (imputed_lst > 220) & (imputed_lst < 350) &
                    np.isfinite(ref_lst) & np.isfinite(imputed_lst)
            )

            # --- 3. Check for sufficient data and train the model ---
            if np.sum(valid_mask) < 1000:  # Minimum threshold for a reliable model
                print(f"  - Skipping: Insufficient valid pixels ({np.sum(valid_mask)}) for {date_str}")
                continue

            ref_valid = ref_lst[valid_mask]
            imputed_valid = imputed_lst[valid_mask]

            # Initialize a new matcher for EACH day
            cdf_matcher = CDFMatcher(n_quantiles=n_quantiles, breakpoints=breakpoints)
            cdf_matcher.add_training_data(ref_valid, imputed_valid)

            print(f"  - Found {len(ref_valid)} valid pixels. Building daily CDF model...")
            cdf_matcher.build_cdf_matching_model()

            # --- 4. Apply the daily model ---
            print("  - Applying daily correction...")
            corrected_lst = cdf_matcher.apply_correction(imputed_lst)

            index = np.where((np.abs(corrected_lst - ref_lst) > 20) & (roi_mask == 1))
            corrected_lst[index] = ref_lst[index]

            corrected_lst[roi_mask != 1] = np.nan  # Apply mask
            corrected_lst[(corrected_lst > 350) | (corrected_lst < 220) ] = np.nan  # Apply mask

            # --- 5. Save the corrected file ---
            output_file = os.path.join(output_dir, f"CDF_Corrected_PMW_LST_{date_str}.tif")
            save_geotiff(output_file, corrected_lst, proj, geotrans, no_data_value=-9999)

            print(f"  - Success! Saved corrected file to {output_file}")
            success_count += 1

        except Exception as e:
            print(f"  - ERROR processing {date_str}: {e}")
            traceback.print_exc()  # Print full error for debugging
            continue

    print(f"\nCDF correction complete! Successfully processed {success_count}/{len(available_files)} files.")


def plot_single_day_cdf_analysis(
        date_str,
        imputed_lst_dir,
        reference_lst_dir,
        roi_file,
        output_dir,
        corrected_lst_dir=None
):
    """Plots a single-day CDF analysis (Unchanged)"""

    print(f"Starting to plot CDF analysis for {date_str}...")

    roi_mask = gdal_array.LoadFile(roi_file)

    datasets = {}

    imputed_file = os.path.join(imputed_lst_dir, f"Imputed_PMW_LST_{date_str}.tif")
    if os.path.exists(imputed_file):
        datasets['Imputed'] = gdal_array.LoadFile(imputed_file)
    else:
        print(f"Imputed LST file not found for: {date_str}")
        return

    ref_file = os.path.join(reference_lst_dir, f"skin_temperature_{date_str}_1015LT_3point.tif")
    if os.path.exists(ref_file):
        datasets['Reference'] = gdal_array.LoadFile(ref_file)
    else:
        print(f"Reference LST file not found for: {date_str}")
        return

    if corrected_lst_dir:
        corrected_file = os.path.join(corrected_lst_dir, f"CDF_Corrected_PMW_LST_{date_str}.tif")
        if os.path.exists(corrected_file):
            datasets['CDF_Corrected'] = gdal_array.LoadFile(corrected_file)

    processed_data = {}
    for name, data in datasets.items():
        if name == 'CDF_Corrected':
            valid_mask = (roi_mask == 1) & (data > 220) & (data < 350) & (data != -9999)
        else:
            valid_mask = (roi_mask == 1) & (data > 220) & (data < 350)

        if np.sum(valid_mask) > 0:
            processed_data[name] = data[valid_mask]
            print(f"{name} data: {len(processed_data[name])} valid pixels")
        else:
            print(f"{name} data: No valid pixels")

    if len(processed_data) < 2:
        print("Insufficient valid data to plot charts")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'LST CDF Analysis for {date_str}', fontsize=16, fontweight='bold')

    colors = {'Reference': 'blue', 'Imputed': 'red', 'CDF_Corrected': 'green'}
    linestyles = {'Reference': '-', 'Imputed': '--', 'CDF_Corrected': '-.'}

    ax1 = axes[0, 0]
    for name, data in processed_data.items():
        sorted_data = np.sort(data)
        cdf_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data) * 100
        ax1.plot(sorted_data, cdf_values, color=colors.get(name), linestyle=linestyles.get(name), linewidth=2,
                 label=f'{name} (n={len(data)})')
    ax1.set_xlabel('Temperature (K)');
    ax1.set_ylabel('CDF (%)');
    ax1.set_title('Cumulative Distribution Function Comparison');
    ax1.grid(True, alpha=0.3);
    ax1.legend();
    ax1.set_xlim(220, 350)

    ax2 = axes[0, 1]
    for name, data in processed_data.items():
        ax2.hist(data, bins=50, alpha=0.6, density=True, color=colors.get(name), label=f'{name}')
    ax2.set_xlabel('Temperature (K)');
    ax2.set_ylabel('Probability Density');
    ax2.set_title('Probability Density Function Comparison');
    ax2.grid(True, alpha=0.3);
    ax2.legend();
    ax2.set_xlim(220, 350)

    ax3 = axes[1, 0]
    percentiles = np.arange(0, 101, 5)
    for name, data in processed_data.items():
        quantiles = np.percentile(data, percentiles)
        ax3.plot(percentiles, quantiles, color=colors.get(name), linestyle=linestyles.get(name), linewidth=2,
                 marker='o', markersize=4, label=name)
    ax3.set_xlabel('Percentile (%)');
    ax3.set_ylabel('Temperature (K)');
    ax3.set_title('Quantile-Quantile Plot');
    ax3.grid(True, alpha=0.3);
    ax3.legend()

    ax4 = axes[1, 1];
    ax4.axis('off')
    stats_data = [];
    col_order = ['Reference', 'Imputed', 'CDF_Corrected'];
    datasets_to_stat = [d for d in col_order if d in processed_data]
    for name in datasets_to_stat:
        data = processed_data[name];
        stats = {'Dataset': name, 'Mean': f'{np.mean(data):.2f}', 'Std': f'{np.std(data):.2f}',
                 'Min': f'{np.min(data):.2f}', 'Max': f'{np.max(data):.2f}', 'Median': f'{np.percentile(data, 50):.2f}',
                 'Count': f'{len(data)}'};
        stats_data.append(list(stats.values()))
    stats_df = pd.DataFrame(stats_data, columns=stats.keys())
    table = ax4.table(cellText=stats_df.values, colLabels=stats_df.columns, cellLoc='center', loc='center',
                      colWidths=[0.2, 0.15, 0.15, 0.15, 0.15, 0.15, 0.2]);
    table.auto_set_font_size(False);
    table.set_fontsize(10);
    table.scale(1, 1.8)
    for (row, col), cell in table.get_celld().items():
        if row == 0: cell.set_facecolor('#40466e'); cell.set_text_props(weight='bold', color='white')
    ax4.set_title('Statistical Summary', fontweight='bold', pad=20)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_file = os.path.join(output_dir, f'LST_CDF_Analysis_{date_str}.png')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"CDF analysis plot saved to: {output_file}")


def plot_example_cdf_analysis():
    """Plots a CDF analysis chart for an example date (Unchanged)"""
    date_str = "20190701"
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()  # Fallback for interactive environments
    project_dir = os.path.dirname(os.path.dirname(script_dir))
    data_root_dir = os.path.join(project_dir, 'Data', 'Sub_to_25km')
    imputed_lst_dir = os.path.join(data_root_dir, 'Imputed_PMW_LST_25km')
    reference_lst_dir = os.path.join(data_root_dir, 'ERA5_25km')
    corrected_lst_dir = os.path.join(data_root_dir, 'CDF_Corrected_PMW_LST_25km')
    roi_file = os.path.join(project_dir, 'Data', 'Base', 'FY3_VIRR_NDVI_MAX_2019_25km_ROI.tif')
    output_dir = os.path.join(data_root_dir, 'CDF_Analysis_Plots')
    plot_single_day_cdf_analysis(date_str=date_str, imputed_lst_dir=imputed_lst_dir,
                                 reference_lst_dir=reference_lst_dir, roi_file=roi_file, output_dir=output_dir,
                                 corrected_lst_dir=corrected_lst_dir)


if __name__ == '__main__':
    YEAR = 2019

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()  # Fallback for interactive environments

    project_dir = os.path.dirname(os.path.dirname(script_dir))
    data_root_dir = os.path.join(project_dir, 'Data', 'Sub_to_25km')
    imputed_lst_dir = os.path.join(data_root_dir, 'Imputed_PMW_LST_25km')
    reference_lst_dir = os.path.join(data_root_dir, 'ERA5_25km')
    roi_file = os.path.join(project_dir, 'Data', 'Base', 'FY3_VIRR_NDVI_MAX_2019_25km_ROI.tif')
    output_dir = os.path.join(data_root_dir, 'CDF_Corrected_PMW_LST_25km')

    try:
        # MODIFIED: Call to the new daily processing function.
        # Note that `sample_days` is no longer needed.
        batch_process_cdf_correction(
            year=YEAR,
            imputed_lst_dir=imputed_lst_dir,
            reference_lst_dir=reference_lst_dir,
            output_dir=output_dir,
            roi_file=roi_file,
           # breakpoints=[260, 310, 330]  # Or use n_quantiles=7
            n_quantiles=12
        )
    except Exception as e:
        print(f"Program execution failed: {e}")
        traceback.print_exc()

    print("\n" + "=" * 50)
    print("Plotting example CDF analysis chart for a single day...")
    plot_example_cdf_analysis()