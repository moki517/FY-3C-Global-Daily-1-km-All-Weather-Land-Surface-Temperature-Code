import os
from osgeo import gdal
import numpy as np
import math
from osgeo import osr

gdal.UseExceptions()


def get_data_list(file_path, out=""):
    list1 = []  # 文件的完整路径
    if os.path.isdir(file_path):
        fileList = os.listdir(file_path)
        # Filter out system files like Thumbs.db if any
        fileList = [f for f in fileList if not f.startswith('.')]
        fileList.sort() # Optional: sort files for consistent processing order
        if out != "":
            for f in fileList:
                # Construct output path, replacing extension or adding one if needed
                # This assumes input files might be .HDF and output should be .tif
                # Adjust if input files have different extensions or output needs a specific name
                base, ext = os.path.splitext(f)
                # Example: replace .HDF with .tif, or keep base name + .tif
                # out_data = os.path.join(out, base + ".tif") # Keep original base name, add .tif
                out_data = os.path.join(out, f) # Default: use original filename
                # out_data = out + "\\" + f # Original logic - less safe path joining
                list1.append(out_data)
        else:
            for f in fileList:
                pre_data = os.path.join(file_path, f)  # 文件的完整路径
                # pre_data = file_path + '\\' + f # Original logic - less safe path joining
                list1.append(pre_data)
        return list1

def H2W_new(infile, outfile, fill_value_para, scale_and_int=False, scale_factor=10):
    """WGS84转换工作函数 - 用于多进程"""
    if os.path.exists(outfile):
        return f"EXISTS: {outfile}"

    # 确保输出目录存在
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    try:
        ds = gdal.Open(infile)
        if ds is None:
            return f"ERROR: Cannot open {infile}"

        ingeo = ds.GetGeoTransform()
        cols = ds.RasterXSize
        rows = ds.RasterYSize
        or_x = ingeo[0]
        or_y = ingeo[3]

        # X和Y方向分块 - 减少块数以提高效率
        xblocksize = max(int((cols + 1) / 3), 1000)  # 减少分块数
        yblocksize = max(int((rows + 1) / 3), 1000)
        lon_max = -360
        lon_min = 360
        lat_max = -90
        lat_min = 90

        # 计算经纬度范围
        for i in range(0, rows + 1, yblocksize):
            if i + yblocksize < rows + 1:
                numrows = yblocksize
            else:
                numrows = rows + 1 - i
            for j in range(0, cols + 1, xblocksize):
                if j + xblocksize < cols + 1:
                    numcols = xblocksize
                else:
                    numcols = cols + 1 - j

                x = ingeo[0] + j * ingeo[1]
                y = ingeo[3] + i * ingeo[5]
                xgrid, ygrid = np.meshgrid(np.linspace(x, x + numcols * ingeo[1], num=numcols),
                                           np.linspace(y, y + numrows * ingeo[5], num=numrows))

                # Hammer坐标转换
                xgrid = np.where(xgrid > (18000.0 * 1000.0), (18000.0 * 1000.0) - xgrid, xgrid)
                xgrid = xgrid / (18000.0 * 1000.0)
                ygrid = np.where(ygrid > (9000.0 * 1000.0), (9000.0 * 1000.0) - ygrid, ygrid)
                ygrid = ygrid / (9000.0 * 1000.0)
                z = np.sqrt(1 - np.square(xgrid) / 2.0 - np.square(ygrid) / 2.0)
                lon = 2 * np.arctan(np.sqrt(2) * xgrid * z / (2.0 * (np.square(z)) - 1))
                lat = np.arcsin(np.sqrt(2) * ygrid * z)

                lon = lon / math.pi * 180.0
                lat = lat / math.pi * 180.0

                lon_max = max(lon_max, np.max(lon))
                lon_min = min(lon_min, np.min(lon))
                lat_max = max(lat_max, np.max(lat))
                lat_min = min(lat_min, np.min(lat))

        newcols = math.ceil((lon_max - lon_min) / 0.01)
        newrows = math.ceil((lat_max - lat_min) / 0.01)

        # 设置数据类型
        if scale_and_int:
            output_dtype = gdal.GDT_Int16
            fill_value = fill_value_para
        else:
            output_dtype = gdal.GDT_Float32
            fill_value = np.nan

        driver = gdal.GetDriverByName("GTiff")
        outds = driver.Create(outfile, newcols, newrows, 1, output_dtype,
                              options=['COMPRESS=LZW', 'PREDICTOR=2', 'TILED=YES', 'BIGTIFF=YES'])  # 添加BIGTIFF支持)
        geo2 = (lon_min, 0.01, 0, lat_max, 0, -1 * 0.01)
        oproj_srs = osr.SpatialReference()
        proj_4 = "+proj=longlat +datum=WGS84 +no_defs"
        oproj_srs.ImportFromProj4(proj_4)
        outds.SetGeoTransform(geo2)
        outds.SetProjection(oproj_srs.ExportToWkt())
        outband = outds.GetRasterBand(1)
        datav = ds.ReadAsArray()

        data = datav.copy()
        # 减少分块数以提高效率
        xblocksize = max(int(newcols / 3), 1000)
        yblocksize = max(int(newrows / 3), 1000)

        # 重采样到WGS84
        for i in range(0, newrows, yblocksize):
            if i + yblocksize < newrows:
                numrows = yblocksize
            else:
                numrows = newrows - i
            for j in range(0, newcols, xblocksize):
                if j + xblocksize < newcols:
                    numcols = xblocksize
                else:
                    numcols = newcols - j

                x = lon_min + j * 0.01 + 0.01 / 2.0
                y = lat_max + i * (-1 * 0.01) - 0.01 / 2.0
                newxgrid, newygrid = np.meshgrid(np.linspace(x, x + numcols * 0.01, num=numcols),
                                                 np.linspace(y, y + numrows * (-1 * 0.01), num=numrows))

                # 经纬度转Hammer坐标
                newxgrid = np.where(newxgrid > 180.0, newxgrid - 360.0, newxgrid)
                newxgrid = newxgrid / 180.0 * math.pi
                newygrid = newygrid / 180.0 * math.pi
                newz = np.sqrt(1 + np.cos(newygrid) * np.cos(newxgrid / 2.0))
                x = np.cos(newygrid) * np.sin(newxgrid / 2.0) / newz
                y = np.sin(newygrid) / newz

                x = x * (18000.0 * 1000.0)
                y = y * (9000.0 * 1000.0)
                x_index = (np.floor((x - or_x) / ingeo[1])).astype(int)
                x_index = np.where(x_index < 0, data.shape[1] - 1, x_index)
                x_index = np.where(x_index >= data.shape[1], data.shape[1] - 1, x_index)
                y_index = (np.floor((y - or_y) / ingeo[5])).astype(int)
                y_index = np.where(y_index < 0, data.shape[0] - 1, y_index)
                y_index = np.where(y_index >= data.shape[0], data.shape[0] - 1, y_index)
                newdata = data[y_index, x_index]

                # 修复：缩放和整型转换
                if scale_and_int:
                    invalid_mask = np.isnan(newdata)
                    valid_mask = ~np.isnan(newdata)
                    newdata_int = np.full(newdata.shape, fill_value_para, dtype=np.int16)
                    newdata_int[valid_mask] = np.round((newdata[valid_mask]*scale_factor)).astype(np.int16)
                    newdata = newdata_int
                    newdata[invalid_mask] = fill_value_para

                    # # 检查原始数据是否有特定的无效值标识
                    # if np.issubdtype(data.dtype, np.floating):
                    #     # 对于浮点型数据，识别NaN值
                    #     valid_mask = ~np.isnan(newdata)
                    #     newdata_int = np.full(newdata.shape, fill_value_para, dtype=np.int16)
                    #     if np.any(valid_mask):
                    #         newdata_int[valid_mask] = np.round(newdata[valid_mask] * 10).astype(np.int16)
                    #     newdata = newdata_int
                    # else:
                    #     # 对于整型数据，需要确定原始的无效值
                    #     # 假设原始数据中的无效值已经被设置为特定值（比如-9999或0）
                    #     # 这里需要根据实际情况调整无效值的判断条件
                    #     original_nodata = ds.GetRasterBand(1).GetNoDataValue()
                    #     if original_nodata is not None:
                    #         valid_mask = newdata != original_nodata
                    #     else:
                    #         # 如果没有设置无效值，假设所有数据都有效
                    #         valid_mask = np.ones(newdata.shape, dtype=bool)
                    #
                    #     newdata_int = np.full(newdata.shape, fill_value_para, dtype=np.int16)
                    #     if np.any(valid_mask):
                    #         newdata_int[valid_mask] = np.round(newdata[valid_mask] * 10).astype(np.int16)
                    #     newdata = newdata_int

                outband.WriteArray(newdata, j, i)

        #设置无效值
        if scale_and_int:
            outband.SetNoDataValue(fill_value_para)
        else:
            outband.SetNoDataValue(np.nan)
        outband.FlushCache()
        outds = None
        ds = None
        return f"SUCCESS: {outfile}"

    except Exception as e:
        return f"ERROR: {outfile} - {str(e)}"

def H2W(infile, outfile, fill_value_para, scale_and_int=False, scale_factor=10):
    """
    将Hammer投影转换为WGS84地理坐标系

    Parameters:
    -----------
    infile : str
        输入文件路径
    outfile : str
        输出文件路径
    fill_value_para : numeric
        指定的输出无效值。如果 scale_and_int=True，此值应为整型。
        如果 scale_and_int=False，此值通常是 np.nan。
    scale_and_int : bool, default=False
        是否将输出值乘以 scale_factor 并转换为整型。
    scale_factor : numeric, default=10
        当 scale_and_int=True 时，用于将数据乘以的因子。
    """
    ds = None # Initialize ds to None
    try:
        ds = gdal.Open(infile)
        if ds is None:
            print(f"Error: Could not open file {infile}")
            return

        ingeo = ds.GetGeoTransform()
        cols = ds.RasterXSize
        rows = ds.RasterYSize
        or_x = ingeo[0]
        or_y = ingeo[3]

        inband = ds.GetRasterBand(1)
        datav = inband.ReadAsArray()
        in_nodata = inband.GetNoDataValue()

        # --- Calculate output bounds by projecting corners ---
        # While projecting all points in blocks gives potentially tighter bounds,
        # projecting corners is simpler and sufficient for many cases.
        # Hammer corners in source projection:
        # (or_x, or_y), (or_x + cols*ingeo[1], or_y), (or_x, or_y + rows*ingeo[5]), (or_x + cols*ingeo[1], or_y + rows*ingeo[5])
        # Let's project the approximate center (or 0,0 in Hammer) which is 0,0 in WGS84
        # The Hammer projection domain is roughly -180E to 180E, -90N to 90N in WGS84,
        # centered around 0,0. The Hammer projection in the source file
        # seems to map to a large area.
        # A simpler approach is to calculate the range of Lats/Lons for the *entire* grid
        # without blocking just for the bounds calculation.

        # Calculate Hammer coordinates for the whole grid
        x_coords = ingeo[0] + np.arange(cols) * ingeo[1] + ingeo[1] / 2.0 # Use pixel centers
        y_coords = ingeo[3] + np.arange(rows) * ingeo[5] + ingeo[5] / 2.0 # Use pixel centers
        hammer_x, hammer_y = np.meshgrid(x_coords, y_coords)

        # Convert Hammer to WGS84 (Same logic as inside block loop)
        # First normalize Hammer coordinates
        # Note: The constants 18000.0 * 1000.0 and 9000.0 * 1000.0
        # seem specific to the input dataset's Hammer units/extent.
        # These should ideally be derived from the source projection definition
        # or metadata if possible, rather than hardcoded.
        # Assuming they represent the extents for normalization:
        hammer_x_norm = hammer_x / (18000.0 * 1000.0)
        hammer_y_norm = hammer_y / (9000.0 * 1000.0)

        # Avoid potential issues with values slightly outside the [-1, 1] range
        hammer_x_norm = np.clip(hammer_x_norm, -1.0, 1.0)
        hammer_y_norm = np.clip(hammer_y_norm, -1.0, 1.0)

        # Hammer Inverse Projection Formula
        z = np.sqrt(1 - np.square(hammer_x_norm) / 2.0 - np.square(hammer_y_norm) / 2.0)
        # Avoid division by zero or near zero if z is small
        # Points where 2*z*z - 1 is zero or negative are likely invalid Hammer points
        # or map to poles/antimeridian where transformation is singular.
        # Let's handle these carefully.
        denominator = 2.0 * np.square(z) - 1
        # For longitude, handle denominator near zero.
        # When denominator is zero, 2*z*z = 1, so z = 1/sqrt(2). This occurs at x_norm^2/2 + y_norm^2/2 = 1/2, i.e., x_norm^2 + y_norm^2 = 1. This is the boundary circle of the Hammer projection. Points on this boundary map to the ±180 meridian.
        # For points where denominator is negative, 2*z*z < 1. This means z < 1/sqrt(2), which implies x_norm^2 + y_norm^2 > 1. These points are outside the valid Hammer projection domain (the ellipse).
        valid_mask_proj = (denominator > 1e-9) # Or some small tolerance, avoid <= 0

        lon_rad = np.full_like(hammer_x_norm, np.nan)
        # Apply transformation only to valid points
        lon_rad[valid_mask_proj] = 2 * np.arctan(np.sqrt(2) * hammer_x_norm[valid_mask_proj] * z[valid_mask_proj] / denominator[valid_mask_proj])

        lat_rad = np.arcsin(np.sqrt(2) * hammer_y_norm * z)
        # Handle potential domain errors for arcsin due to float precision
        lat_rad = np.clip(lat_rad, -math.pi/2, math.pi/2)

        lon_deg = np.degrees(lon_rad)
        lat_deg = np.degrees(lat_rad)

        # Find min/max lat/lon from the projected valid points
        # Filter out NaNs before finding min/max
        # Also filter out points where the projection calculation failed
        valid_lon = lon_deg[valid_mask_proj & ~np.isnan(lon_deg) & ~np.isnan(lat_deg)]
        valid_lat = lat_deg[valid_mask_proj & ~np.isnan(lon_deg) & ~np.isnan(lat_deg)]


        if valid_lon.size == 0 or valid_lat.size == 0:
             print(f"Warning: No valid geographic coordinates found for {infile}. Skipping.")
             ds = None # Ensure dataset is closed
             return

        lon_max = np.max(valid_lon)
        lon_min = np.min(valid_lon)
        lat_max = np.max(valid_lat)
        lat_min = np.min(valid_lat)

        hammer_x, hammer_y, hammer_x_norm, hammer_y_norm, z, lon_rad, lat_rad, lon_deg, lat_deg, valid_mask_proj = None, None, None, None, None, None, None, None, None, None # Release memory


        # Determine output resolution (assuming 0.01 degree is desired)
        output_res = 0.01 # Use 0.0025 for 250m equivalent at equator

        newcols = math.ceil((lon_max - lon_min) / output_res)
        newrows = math.ceil((lat_max - lat_min) / output_res)

        # Adjust origin slightly to align pixel centers if needed
        # The geo transform assumes the top-left corner of the top-left pixel.
        # For pixel center calculation, origin is (lon_min, lat_max)
        output_origin_x = lon_min
        output_origin_y = lat_max
        output_geotransform = (output_origin_x, output_res, 0, output_origin_y, 0, -output_res)

        # 根据是否需要整型输出选择数据类型和输出无效值
        if scale_and_int:
            output_dtype = gdal.GDT_Int16  # 使用16位整型
            output_fill_value = fill_value_para # 使用传入的整型无效值
            # Ensure fill_value_para is within Int16 range if scale_and_int is True
            if not np.issubdtype(type(output_fill_value), np.integer) or output_fill_value < -32768 or output_fill_value > 32767:
                 print(f"Warning: Specified fill_value_para ({fill_value_para}) is not a valid Int16 value when scale_and_int is True. Using -32768 instead.")
                 output_fill_value = -32768
            # Ensure scale_factor is numeric
            if not isinstance(scale_factor, (int, float)):
                 print(f"Warning: Specified scale_factor ({scale_factor}) is not numeric. Using default 10.")
                 scale_factor = 10.0 # Use float for calculation
            else:
                 scale_factor = float(scale_factor) # Ensure float for calculation

        else:
            output_dtype = gdal.GDT_Float32  # 使用32位浮点型
            output_fill_value = np.nan  # 使用np.nan作为浮点型无效值


        driver = gdal.GetDriverByName("GTiff")
        # Ensure output directory exists
        outdir = os.path.dirname(outfile)
        if outdir and not os.path.exists(outdir):
            os.makedirs(outdir)

        outds = None # Initialize outds to None
        try:
            outds = driver.Create(outfile, newcols, newrows, 1, output_dtype)
            if outds is None:
                print(f"Error: Could not create output file {outfile}")
                ds = None # Ensure input is closed
                return

            oproj_srs = osr.SpatialReference()
            proj_4 = "+proj=longlat +datum=WGS84 +no_defs"
            oproj_srs.ImportFromProj4(proj_4)
            outds.SetGeoTransform(output_geotransform)
            outds.SetProjection(oproj_srs.ExportToWkt())
            outband = outds.GetRasterBand(1)

            # Set the output nodata value on the band
            if np.isnan(output_fill_value):
                 outband.SetNoDataValue(output_fill_value)
            else:
                 outband.SetNoDataValue(float(output_fill_value)) # Ensure it's set as a float even if int

            # --- Create a temporary array to hold input data + edge fill ---
            # This temporary array should preserve original data precision or use float32
            # Initialize with np.nan to represent invalid/out-of-bounds points
            # Size is (rows+1, cols+1) to safely handle indices equal to original dimension size
            temp_data = np.full((rows + 1, cols + 1), np.nan, dtype=np.float32)

            # Copy original data into the temporary array
            # If input nodata exists, convert it to np.nan in the temporary array
            if in_nodata is not None:
                 temp_data[0:rows, 0:cols] = np.where(datav == in_nodata, np.nan, datav)
            else:
                 # Handle potential non-float input datav type by casting if necessary
                 if np.issubdtype(datav.dtype, np.integer):
                     # Simple integer copy, NaNs handled by initialization
                     temp_data[0:rows, 0:cols] = datav.astype(np.float32) # Cast int data to float for consistency
                 else:
                     temp_data[0:rows, 0:cols] = datav # Direct copy for float/double


            datav = None # Release memory from original datav array

            # --- Reproject pixel by pixel using nearest neighbor ---
            # Process output in blocks
            xblocksize = int(newcols / 5) if newcols > 5 else newcols
            yblocksize = int(newrows / 5) if newrows > 5 else newrows
            if xblocksize == 0: xblocksize = 1
            if yblocksize == 0: yblocksize = 1


            for i in range(0, newrows, yblocksize):
                numrows = min(yblocksize, newrows - i) # Use min to handle last block
                for j in range(0, newcols, xblocksize):
                    numcols = min(xblocksize, newcols - j) # Use min to handle last block

                    # Calculate WGS84 coordinates for output block pixel centers
                    # Using the output_geotransform is safer:
                    x_block_coords = output_geotransform[0] + (j + np.arange(numcols)) * output_geotransform[1] + output_geotransform[1] / 2.0
                    y_block_coords = output_geotransform[3] + (i + np.arange(numrows)) * output_geotransform[5] + output_geotransform[5] / 2.0

                    newxgrid, newygrid = np.meshgrid(x_block_coords, y_block_coords)

                    # --- Convert WGS84 (Lon/Lat) to Hammer ---
                    # Keep longitude in -180 to 180 range for conversion formula
                    newxgrid_rad = np.radians(newxgrid) # Lon in radians
                    newygrid_rad = np.radians(newygrid) # Lat in radians
                    newxgrid, newygrid = None, None # Release memory

                    # Hammer forward projection formula
                    newz = np.sqrt(1 + np.cos(newygrid_rad) * np.cos(newxgrid_rad / 2.0))
                    # Avoid division by zero or near zero for hammer conversion as well
                    valid_mask_hammer_fwd = (newz > 1e-9) # Avoid division by zero z
                    hammer_x_calc = np.full_like(newxgrid_rad, np.nan)
                    hammer_y_calc = np.full_like(newygrid_rad, np.nan)

                    hammer_x_calc[valid_mask_hammer_fwd] = (np.cos(newygrid_rad[valid_mask_hammer_fwd]) * np.sin(newxgrid_rad[valid_mask_hammer_fwd] / 2.0) / newz[valid_mask_hammer_fwd]) * (18000.0 * 1000.0) # Convert back to input units
                    hammer_y_calc[valid_mask_hammer_fwd] = (np.sin(newygrid_rad[valid_mask_hammer_fwd]) / newz[valid_mask_hammer_fwd]) * (9000.0 * 1000.0) # Convert back to input units

                    newxgrid_rad, newygrid_rad, newz, valid_mask_hammer_fwd = None, None, None, None # Release memory


                    # --- Calculate Input Pixel Indices (Nearest Neighbor) ---
                    # Indices relative to the *original* datav array (0 to rows-1, 0 to cols-1)
                    # Using floor on the calculated hammer coordinates relative to the input geotransform
                    # Note: ingeo[1] is pixel width (positive), ingeo[5] is pixel height (negative)
                    # Need to be careful with the sign of ingeo[5]
                    x_index_float = (hammer_x_calc - ingeo[0]) / ingeo[1]
                    y_index_float = (hammer_y_calc - ingeo[3]) / ingeo[5] # This naturally handles the negative resolution

                    # Use np.round for nearest neighbor
                    x_index = np.round(x_index_float).astype(int)
                    y_index = np.round(y_index_float).astype(int)

                    hammer_x_calc, hammer_y_calc, x_index_float, y_index_float = None, None, None, None # Release memory

                    # --- Sample data from the temporary array ---
                    # Handle out-of-bounds indices by assigning NaN.
                    # Indices must be within [0, cols-1] and [0, rows-1] for the original data part of temp_data.

                    # Initialize sampled data block with NaN (matches temp_data's nodata representation)
                    sampled_data = np.full((numrows, numcols), np.nan, dtype=np.float32)

                    # Create a mask for indices that are within the original data bounds [0, cols-1] and [0, rows-1]
                    valid_indices_mask = (x_index >= 0) & (x_index < cols) & \
                                         (y_index >= 0) & (y_index < rows)

                    # Use the valid indices to sample from the part of temp_data that holds original data
                    # Need flattened indices for sampling `temp_data` correctly if x_index, y_index are meshgrids
                    # Let's use the boolean mask directly on the original meshgrid-shaped indices
                    sampled_data[valid_indices_mask] = temp_data[y_index[valid_indices_mask], x_index[valid_indices_mask]]

                    x_index, y_index = None, None # Release memory

                    # --- Process sampled data based on output type ---
                    data_to_write = sampled_data # Start with the float32 sampled data

                    if scale_and_int:
                        # Output needs to be Int16 with scaling and the integer fill_value_para
                        # Create an output block array initialized with the integer fill value
                        output_block = np.full_like(sampled_data, output_fill_value, dtype=np.int16)

                        # Identify valid data in the sampled float data (not NaN)
                        valid_mask = ~np.isnan(sampled_data)

                        # Apply scaling, rounding, and casting to Int16 *only* for valid data
                        # Use np.round to get nearest integer before casting
                        output_block[valid_mask] = np.round(sampled_data[valid_mask] * scale_factor).astype(np.int16)

                        data_to_write = output_block # Use the processed integer data

                    # Write the processed block to the output band
                    outband.WriteArray(data_to_write, j, i)

            # --- Cleanup ---
            outband.FlushCache()
            outband = None
            outds = None # Close output dataset (flushes changes to disk)

        except Exception as e:
            print(f"An error occurred processing {infile}: {e}")
            # Clean up output file if creation failed or error occurred during writing
            if outds:
                 outds = None
            if os.path.exists(outfile):
                 try:
                     # Check if file size is minimal (indicating failed write)
                     if os.path.getsize(outfile) < 1000: # Arbitrary small size threshold
                          os.remove(outfile)
                          print(f"Removed potentially incomplete output file: {outfile}")
                 except Exception as remove_e:
                     print(f"Could not remove incomplete output file {outfile}: {remove_e}")
        finally:
            if ds:
                ds = None # Close input dataset


    except Exception as e:
        print(f"An error occurred opening or initializing processing for {infile}: {e}")
    finally:
        if ds: # Ensure input dataset is closed even if opening succeeded but later steps failed
            ds = None


if __name__ == '__main__':
    # --- Configuration ---
    infile_path = r"H:\Global_VIRR_LST\temp_mosaic"
    outfile_path = r"f:\FY3C_2019\VIRR_LST"

    # New options:
    SCALE_AND_INT = False  # Set to True to scale and convert to Int16 output
    OUTPUT_SCALE_FACTOR = 10 # Set the scaling factor when SCALE_AND_INT is True

    # Set output nodata value. If scale_and_int=True, it should be an integer.
    # Example: For COT product, original nodata might be 65535 or 255 etc.
    # If outputting Int16 with scaling, choose an Int16 value outside the expected data range, e.g., -32768.
    # If not scaling (Float32 output), np.nan is typically used.
    if SCALE_AND_INT:
        # Choose an integer value valid for Int16 (-32768 to 32767)
        output_nodata_value = 0 # Example: use -9999 for Int16 output scaled by 100
    else:
        output_nodata_value = np.nan # Example: use np.nan for Float32 output

    # --- Processing ---
    infile_list = get_data_list(infile_path)
    # Corrected way to generate outfile_list to ensure correct extensions
    # Assuming input is HDF and output should be TIF
    outfile_list_corrected = []
    for infile in infile_list:
        base_name = os.path.basename(infile)
        name_without_ext, _ = os.path.splitext(base_name)
        output_name = name_without_ext + ".tif" # Or any desired output extension
        outfile_path_full = os.path.join(outfile_path, output_name)
        outfile_list_corrected.append(outfile_path_full)

    from tqdm import tqdm

    # Ensure lists have the same length
    if len(infile_list) != len(outfile_list_corrected):
        print("Error: Input and output file lists have different lengths.")
    else:
        for infile, outfile in tqdm(zip(infile_list, outfile_list_corrected), desc="Processing files", total=len(infile_list)):
            if os.path.exists(outfile):
                print(f"{outfile} is exists! Skipping.")
                continue
            # Pass the configured parameters, including scale_factor

            H2W_new(infile, outfile, output_nodata_value, scale_and_int=SCALE_AND_INT, scale_factor=OUTPUT_SCALE_FACTOR)