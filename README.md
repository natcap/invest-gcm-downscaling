# User Guide: InVEST GCM Downscaling
This is an InVEST Plugin based on the gcm-downscaling GitHub repo (https://github.com/natcap/gcm-downscaling). This model downscales gridded precipitation data from CMIP6 (Coupled Model Intercomparison Project) GCMs (General Circulation Models) using observed historical precipitation patterns. The goal is to produce realistic future daily precipitation data that reflect both climate model projections and statistical patterns of observed historical records. The model supports both hindcasts and future projections.

## Model Overview
The model bootstraps dates from observed data based on wet/dry/very-wet precipitation transitions and synthesizes projected extremes using generalized Pareto distribution (GPD). It reshuffles observed precipitation to simulate future periods while adjusting extremes based on GCM-projected changes.

## Sources
This plugin relies on an adaptation of the GCMClimTool Library by Angarita H., Yates D., Depsky N. 2014-2021.

Downscaling methods are based on those described in:
- A Technique for Generating Regional Climate Scenarios Using a Nearest-Neighbour Algorithm [http://dx.doi.org/10.1029/2002WR001769]
- Statistical downscaling using K-nearest neighbors [https://doi.org/10.1029/2004WR003444]

Data Sources: ?

## Usage
1.	First, install this plugin. The easiest way to do this is via the InVEST Workbench (version 3.16.0 or later). You can download and install this plugin using its git URL (https://github.com/claire-simpson/invest-gcm-downscaling.git), or, if you prefer, you can clone this repo to your computer and then install it using the path to your local copy.
2.	Once the plugin has been installed, you can run it from the Workbench, just as you would run any InVEST model.

## Data
A datastack json file is provided in this repo along with a sample vector for example/testing purposes only. Analysis-ready data are stored in `zarr` format in a public google cloud bucket (`natcap-climate-data`) in the NatCap Servers cloud project. These data are accessed directly by the model (i.e., you do not need to manually acquire any additional data beyond an AOI vector). Note that no google authentication is needed as these climate data are publicly available.

## Model Inputs & Outputs

### Required Inputs:
- **Area of Interest** (str): Path to a polygon vector (e.g., shapefile) defining the Area of Interest (AOI). Coordinates must be in decimal degrees (WGS84).
- **Workspace Directory** (str): Directory for storing output files and intermediate results.
- **Reference Period Start Date** (str): Start date of the reference period for calculating precipitation normals. Format: 'YYYY-MM-DD'.
- **Reference Period End Date** (str): End date of the reference period for calculating precipitation normals. Format: 'YYYY-MM-DD'.
- **Lower Precipitation Threshold** (float): Lower boundary (in mm) of the middle precipitation state (used for classifying daily states).
- **Upper Precipitation Percentile** (float): Upper boundary (percentile, 0-100) of the middle precipitation state. The corresponding value is computed from observed data.
- **Hindcast** (bool): If True, the model operates in hindcast mode using historical MSWEP data and date range.
### Conditional Inputs (Required if `hindcast=False`):
- **Prediction Start Date** (str): Start date of the forecast period. Format: 'YYYY-MM-DD’.
- **Prediction End Date** (str): End date of the forecast period. Format: 'YYYY-MM-DD’.
- **GCM Model** (str): Name of CMIP6 GCM model to use for projections. 
### Optional Inputs:
- **Observed Dataset Path** (str): Path to an observed, historical precipitation dataset to use instead of the default MSWEP dataset (if hindcasting). The dataset should be a netCDF or other `xarray.open_dataset` readable format. It should contain coordinates and variables named and defined as:
      
      `Coordinates`:
      
      - `lat` - decimal degrees (-90 : 90)
      - `lon` - decimal degrees (-180 : 180) or (0 : 360)
      - `time` - daily timesteps in units that can be parsed to `numpy.datetime64`
        
      `Variables`:
      - `precipitation` - dimensions: (time, lat, lon); units: millimeter
  
- **n_workers** (int): Number of worker processes. If not specified, the model runs in the current process.

### Outputs
*For Each Model & Experiment:*
- **downscaled_precip_[model]_[experiment].nc**: Downscaled precipitation netCDF for the specified GCM model and SSP experiment.
- **downscaled_precip_[model]_[experiment].pdf**: Report PDF containing time series and analysis graphs for the downscaled dataset.

*For Hindcast (if hindcast=True):*
- **downscaled_precip_hindcast.nc**: NetCDF of hindcast precipitation, based on MSWEP alone.
- **downscaled_precip_hindcast.pdf**: Report of the hindcast.
- **bootstrapped_dates_precip_hindcast.csv**: Bootstrapped dates for the hindcast period.

*Intermediate Files (for debugging and reference):*
- **intermediate/extracted_*.nc**: GCM or MSWEP data cropped to AOI.
- **intermediate/aoi_mask_*.nc**: Boolean mask defining pixels inside the AOI.
- **intermediate/*_mean.nc**: Mean values of climate variables within AOI.
- **intermediate/bootstrapped_dates_precip_[model]_[experiment].csv**: CSV of historical dates matched to forecast dates.
- **intermediate/sythesized_extreme_precip_[model]_[experiment].csv**: GPD-sampled historical and forecast extreme precipitation values.

## Notes
- **Experiments**: All experiments (`ssp119`, `ssp126`, `ssp245`, `ssp370`, `ssp460`, `ssp585`) are used for the model selected.
- **Time Handling**: Forecast dates must fall within GCM timespans, with a 15-day buffer at start and end. Reference periods must fall within MSWEP or custom observed data.
- **Extreme Precipitation**: Adjustments are only applied for values exceeding the historical 98th percentile using GPD.
- **Precipitation Units**:
    - GCM: kg/m^2/s → converted to mm/day
    - Observed: mm
- **Calendar**: Uses `noleap` (365-day years) for consistency.
- **Parallelization**: Tasks are managed using taskgraph for reproducibility and performance.

## Troubleshooting
- If GCM files are not found, ensure they are correctly stored in the GCS bucket and readable with anonymous credentials.
- Errors about date bounds typically mean the requested reference or prediction period is outside the range of available data.
- If you see unexpected results in extreme precipitation adjustment, check the `.csv` files created in intermediate/ for GPD samples.

For further support, see the original model repository (https://github.com/natcap/gcm-downscaling).

