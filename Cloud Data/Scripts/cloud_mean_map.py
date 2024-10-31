import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cftime
import warnings
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Define the start and end years for the comparison as cftime objects (NoLeap)
year_start = slice(cftime.DatetimeNoLeap(2020, 1, 1), cftime.DatetimeNoLeap(2040, 12, 31))
year_end = slice(cftime.DatetimeNoLeap(2080, 1, 1), cftime.DatetimeNoLeap(2100, 12, 31))

# Model directories for each variable
model_dirs = {
    'CMCC': 'CMCC/',
    'CNRM': 'CNRM/',
    'EC_Earth': 'EC_Earth/',
    'GFDL': 'GFDL/',
    'IPSL': 'IPSL/',
    'MIROC': 'MIROC/',
    'MPI': 'MPI/'
}

variables = ['clt', 'aermon']  # Variables to process

# Function to load the regridded data and ensure consistent time type
def load_data(model, variable):
    file_path = f"{model_dirs[model]}{variable}/regridded_{variable}_{model}_2015-2100_renamed.nc"
    try:
        ds = xr.open_dataset(file_path, engine='netcdf4')
        
        # Convert time to cftime.DatetimeNoLeap if it's a standard datetime
        if isinstance(ds['time'].values[0], np.datetime64):
            ds['time'] = xr.cftime_range(
                start=pd.Timestamp(ds['time'].values[0]).strftime('%Y-%m-%d'),
                periods=len(ds['time']),
                freq='M',
                calendar='noleap'
            )
        
        return ds
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

# Function to calculate the difference between two periods
def calculate_difference(data, variable):
    time_index = data['time'].values
    if isinstance(time_index[0], cftime.DatetimeNoLeap):
        period_1 = data.sel(time=year_start).mean(dim='time')
        period_2 = data.sel(time=year_end).mean(dim='time')
    else:
        period_1 = data.sel(time=slice('2020-01-01', '2040-12-31')).mean(dim='time')
        period_2 = data.sel(time=slice('2080-01-01', '2100-12-31')).mean(dim='time')
    
    diff = period_2 - period_1
    return diff

# Function to calculate and plot multi-model mean differences with significance
def plot_multi_model_mean(variable, output_file):
    print(f"Starting mean plot for variable: {variable}")
    
    # Load data for each model and append to a list
    datasets = []
    for model in model_dirs.keys():
        data = load_data(model, variable)
        if data is not None:
            diff = calculate_difference(data[variable], variable)
            datasets.append(diff)

    # Check if datasets list is empty
    if not datasets:
        print("No data loaded. Exiting.")
        return

    # Stack datasets along 'model' dimension
    stacked_data = xr.concat(datasets, dim='model')

    # Calculate mean and standard deviation across models
    mean_diff = stacked_data.mean(dim='model')
    std_dev = stacked_data.std(dim='model')

    # Compute threshold for statistical significance
    threshold = 1.96 * (std_dev / np.sqrt(len(datasets)))
    significance_mask = (mean_diff > threshold) | (mean_diff < -threshold)

    # Define longitude and latitude for plotting
    lon = datasets[0]['lon'].values
    lat = datasets[0]['lat'].values

    # Set vmin and vmax based on the variable
    if variable == 'aermon':
        vmin, vmax = -0.05, 0.05
    else:  # Assuming variable is 'clt'
        vmin, vmax = -8,8

    # Plot the mean difference with significance hatching
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': ccrs.Robinson(central_longitude=0)})
    ax.set_global()
    ax.coastlines(resolution='110m', linewidth=1)
    ax.gridlines(linestyle=':')

    # Plot the mean difference
    im = ax.pcolormesh(lon, lat, mean_diff, cmap='RdBu_r', transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)

    # Add contour hatching for significance
    ax.contourf(lon, lat, significance_mask, levels=[0.5, 1.5], colors='none', hatches=['///'], transform=ccrs.PlateCarree())

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', shrink=0.5, pad=0.05)
    cbar_label = 'Aer od500nm' if variable == 'aermon' else variable.upper()
    cbar.set_label(cbar_label, fontsize=12)

    # Save plot
    plt.title(f'Mean {variable.upper()} Difference and Significance (2020-2040 vs 2080-2100)', fontsize=16)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Mean {variable.upper()} difference plot saved to {output_file}\n")

# Generate plots for both variables
for var in variables:
    plot_multi_model_mean(var, f'mean_diff_significance_{var}_2020-2040_vs_2080-2100.png')
