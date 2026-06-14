# CAS-LST Framework

**A Cascading Spatiotemporal Reconstruction Framework for Generating Global 1-km All-Weather Land Surface Temperature from FY-3C**

[![Python Version](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![DOI: Toy Dataset](https://img.shields.io/badge/DOI-10.5281/zenodo.20691572-green.svg)](https://doi.org/10.5281/zenodo.20691572)
[![DOI: Output Data](https://img.shields.io/badge/DOI-10.5281/zenodo.20471030-green.svg)](https://doi.org/10.5281/zenodo.20471030)

## Overview

This repository contains the complete source code and processing scripts for the **CAS-LST** framework. CAS-LST is designed to generate global, daily, 1-kilometer resolution, all-weather Land Surface Temperature (LST) products based on observations from the Chinese Fengyun-3C (FY-3C) satellite. 

The framework sequentially integrates ERA5-constrained microwave background reconstruction, Cumulative Distribution Function (CDF) matching for bias correction, Geographically Weighted Regression (GWR) for spatial downscaling, and a machine learning-based (XGBoost) final fusion module.

## Notes on the Sample Dataset for Reviewers

To facilitate a seamless peer-review process and immediate reproducibility without the computational burden of massive raw data preprocessing, we provide a **pre-processed Sample Dataset** via Zenodo ([10.5281/zenodo.20691572](https://doi.org/10.5281/zenodo.20691572)).

> **Important:** This sample dataset contains the **already gap-filled PMW LST intermediate products**. Therefore, you can bypass the complex initial raw HDF5 parsing and ERA5-based gap-filling steps. You can use this dataset to directly test and verify the core structural algorithms of the CAS-LST framework out-of-the-box, starting from the **CDF matching (`CDF_matching.py`)**, followed by **GWR spatial downscaling**, and concluding with the **XGBoost-based LST fusion**.

*(Note: The final high-accuracy products reported in our manuscript were naturally trained using the complete multi-year, multi-scale global datasets.)*

## Requirements

The framework is implemented in **Python 3.9**. The primary dependencies include:

* `gdal` (for geospatial raster operations)
* `xgboost` (for machine learning fusion)
* `mgwr` / `spglm` (for spatial statistics and GWR)
* `scipy`, `numpy`, `pandas` (for array manipulation and statistics)
* `matplotlib` (for CDF validation plotting)

## Repository Structure & Pipeline Execution

The codebase is modularized into four sequential processing stages:

### `1_FY3C_Data_Preprocessing/`
Contains scripts for initial data preparation.
* Parses raw FY-3C VIRR and MWRI HDF5 files to GeoTIFF.
* Performs coordinate reprojection (Hammer to WGS84) and clipping.
* Includes scripts for NDVI smoothing (Savitzky-Golay / SSA).

### `2_PMW_LST_Reconstruction/`
* `PMW_LST_Reconstruction_OnlyERA5_Dekad.py`: Fills orbital gaps in PMW LST using ERA5 constraints.
* `CDF_matching.py`: **[Start here for the Toy Dataset]** Corrects the variance compression in the reconstructed PMW baseline to match the thermal infrared distribution.
* `CDF_Validation.py`: Validates the CDF-corrected output.

### `3_PMW_LST_Downscale/`
* `4.PMW_LST_Downscale_GWR_Bilinear.py`: Spatially downscales the 25-km microwave thermal baseline to a 1-km grid using Geographically Weighted Regression (GWR) based on topography (DEM) and vegetation (NDVI).

### `4_PMW_VIRR_LST_Fusion/`
* `PMW_VIRR_LST_Model_Final.py`: Trains the XGBoost model utilizing multi-source factors with strict physical quality control (ERA5 serves as a baseline to zero out optical anomalies).
* `PMW_VIRR_LST_Fusion_Final.py`: Applies the trained model and residuals to fuse VIRR and PMW observations, generating the final 1-km all-weather LST map.

## Software and Data Availability

* **Original Input Data:** Raw FY-3C data are openly provided by the National Satellite Meteorological Center (NSMC) of China (http://data.nsmc.org.cn). Ground validation data (for independent validation only) are available from AmeriFlux (https://ameriflux.lbl.gov/) and TPDC (http://data.tpdc.ac.cn).
* **Sample Data for Local Execution:** Available at [Zenodo (10.5281/zenodo.20691572)](https://doi.org/10.5281/zenodo.20691572).
* **Generated LST Products:** The output global daily 1-km LST dataset (3-month sample) is hosted at [Zenodo (10.5281/zenodo.20471030)](https://doi.org/10.5281/zenodo.20471030). The complete long-term dataset is available upon reasonable request.

## Developers & Contact
* **Yuqi Gu** (`moki.gu@foxmail.com`)
* **Qifeng Zhuang** (`zhuangqf@njtech.edu.cn`) 
* School of Geomatics Science and Technology, Nanjing Tech University.
