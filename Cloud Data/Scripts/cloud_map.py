import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cftime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Define the start and end years for the comparison as cftime objects (NoLeap)
year_start = slice(cftime.DatetimeNoLeap(2020, 1, 1), cftime.DatetimeNoLeap(2040, 12, 31))
year_end = slice(cftime.DatetimeNoLeap(2080, 1, 1), cftime.DatetimeNoLeap(2100, 12, 31))

# Models and their respective file paths for clt and aermon
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

# Function to load the regridded data for each model and variable using the renamed files
def load_data(model, variable):
    file = f"{model}/{variable}/regridded_{variable}_{model}_2015-2100_filled.nc"
    try:
        ds = xr.open_dataset(file, engine='netcdf4')
        return ds
    except FileNotFoundError:
        print(f"File not found: {file}")
        return None

# Function to calculate the difference between two periods and handle different time formats
def calculate_difference(data, var):
    time_index = data['time'].values
    if isinstance(time_index[0], cftime.DatetimeNoLeap):
        period_1 = data.sel(time=year_start).mean(dim='time')
        period_2 = data.sel(time=year_end).mean(dim='time')
    else:
        period_1 = data.sel(time=slice('2020-01-01', '2040-12-31')).mean(dim='time')
        period_2 = data.sel(time=slice('2080-01-01', '2100-12-31')).mean(dim='time')
    
    diff = period_2 - period_1
    return diff

# Function to calculate statistical significance for a single model
def calculate_significance_for_model(diff, std_dev):
    threshold = 1.96 * (std_dev / np.sqrt(30))  # Assuming 30 years of data
    significance_mask = (diff > threshold) | (diff < -threshold)
    return significance_mask

# Function to plot the differences with significance and write output to .txt
def plot_model_differences(variable, output_file, txt_file):
    print(f"Starting plot for variable: {variable}")
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 12),
                             subplot_kw={'projection': ccrs.Robinson()})
    axes = axes.flatten()

    for i, (model, folder) in enumerate(model_dirs.items()):
        print(f"Processing model: {model}")
        ax = axes[i]
        data = load_data(model, variable)
        if data is None:
            print(f"Skipping plotting for Model: {model}, Variable: {variable} due to missing data.\n")
            continue
        
        diff = calculate_difference(data[variable], variable)  # Ensure we select the variable (DataArray)
        lon = data['lon'].values
        lat = data['lat'].values
        
        # Calculate standard deviation for significance testing
        std_dev = diff.std()

        # Calculate the significance mask for the current model
        significance_mask = calculate_significance_for_model(diff, std_dev)
        
        ax.set_global()
        ax.coastlines(resolution='110m', linewidth=1)
        ax.gridlines(draw_labels=False, linestyle='--')

        # Set vmin and vmax based on the variable
        if variable == 'aermon':
            vmin, vmax = -0.2, 0.2
        else:
            vmin, vmax = -10, 10

        im = ax.pcolormesh(lon, lat, diff.values, cmap='RdBu_r', transform=ccrs.PlateCarree(),
                           vmin=vmin, vmax=vmax, shading='auto')

        # Apply hatching for statistical significance on both clt and aermon
        hatch_mask = significance_mask.values
        hatch_mask_numeric = np.where(hatch_mask, 1, np.nan)  # Hatch where significant
        ax.contourf(lon, lat, hatch_mask_numeric, levels=[0.5, 1.5], colors='none',
                    hatches=['///'], transform=ccrs.PlateCarree())

        # Set title, handling 'aermon' variable name change
        if variable == 'aermon':
            ax.set_title(f"{model} Aer od500nm", fontsize=12)
        else:
            ax.set_title(f"{model} {variable}", fontsize=12)
        
        print(f"Model {model} plotted with significance hatching\n")

    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Colorbar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal', shrink=0.5, pad=0.1)
    if variable == 'aermon':
        cbar.set_label(f'Aer od500nm', fontsize=12)
    else:
        cbar.set_label(f'{variable.upper()}', fontsize=12)  # Removed extra subtitle for clt

    # Save the plot
    plt.suptitle(f'{variable.upper()} 2080-2100 vs 2020-2040', fontsize=16)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{variable.upper()} plot with significance hatching saved to {output_file}\n")

# First, print the dataset information
print("Starting processing of datasets...\n")

# Generate and save the plots and output the variable/year/difference to a .txt file
with open("variable_differences.txt", "w") as txt_file:
    for var in variables:
        print(f"Starting processing for variable: {var}\n")
        plot_model_differences(var, f'{var}_2080_2100_vs_2020_2040.png', txt_file)
        print(f"Finished processing for variable: {var}\n")
