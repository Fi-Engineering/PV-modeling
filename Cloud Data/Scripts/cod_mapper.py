import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cftime

# Define the start and end years for the comparison as cftime objects (NoLeap)
year_start = slice(cftime.DatetimeNoLeap(2020, 1, 1), cftime.DatetimeNoLeap(2040, 12, 31))
year_end = slice(cftime.DatetimeNoLeap(2080, 1, 1), cftime.DatetimeNoLeap(2100, 12, 31))

# Define the path to the cod file and variable
file_path = 'CNRM/cod/cod_CNRM_2015-2100.nc'
variable = 'cod'  # Adjust this if the variable is named differently in the dataset

# Function to load the cod data
def load_data(file):
    ds = xr.open_dataset(file, engine='netcdf4')
    return ds

# Function to calculate the difference between two periods and return the result
def calculate_difference(data, var):
    print(f"Calculating difference for variable: {var}")

    # Ensure that the time index is properly recognized as cftime.DatetimeNoLeap
    time_index = data['time'].values
    if isinstance(time_index[0], cftime.DatetimeNoLeap):
        period_1 = data.sel(time=year_start).mean(dim='time')
        period_2 = data.sel(time=year_end).mean(dim='time')
    else:
        period_1 = data.sel(time=slice('2020-01-01', '2040-12-31')).mean(dim='time')
        period_2 = data.sel(time=slice('2080-01-01', '2100-12-31')).mean(dim='time')

    diff = period_2 - period_1
    print(f"Difference calculated for variable: {var}")
    return diff

# Function to calculate significance based on a threshold (1.96 sigma)
def calculate_significance(data, threshold=1.96):
    std_dev = data.std(dim='time')
    significance_mask = (data.mean(dim='time') > threshold * std_dev) | (data.mean(dim='time') < -threshold * std_dev)
    return significance_mask

# Function to plot the differences and add hatching for non-significance
def plot_difference(variable, diff, significance_mask, output_file):
    print(f"Starting plot for variable: {variable}")
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.Robinson()})

    lon = diff['lon'].values
    lat = diff['lat'].values

    ax.set_global()
    ax.coastlines(resolution='110m', linewidth=1)
    ax.gridlines(draw_labels=False, linestyle='--')

    # Plot the difference with a color scale
    im = ax.pcolormesh(lon, lat, diff.values, cmap='RdBu_r', transform=ccrs.PlateCarree(), vmin=-5, vmax=5, shading='auto')

    # Add hatching for regions that are NOT statistically significant
    # Here, we invert the significance mask to show non-significant areas
    non_significant_mask = ~significance_mask
    ax.contourf(lon, lat, non_significant_mask.values, levels=[0.5, 1.5], colors='none', hatches=['//'], transform=ccrs.PlateCarree())

    ax.set_title(f"CNRM {variable.upper()} (Non-Sig Hatched)", fontsize=12)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', shrink=0.5, pad=0.1)
    cbar.set_label(f'2080-2100 vs 2020-2040', fontsize=12)

    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_file}")

# Load the dataset
data = load_data(file_path)

# Calculate the difference for the specified variable
difference = calculate_difference(data[variable], variable)

# Calculate the significance mask (ensure data is reduced to 2D for plotting)
significance_mask = calculate_significance(data[variable])

# Plot the difference with hatching only for non-significant areas and save the output
output_file = f'cod_2080_2100_vs_2020_2040_non_sig.png'
plot_difference(variable, difference, significance_mask, output_file)
