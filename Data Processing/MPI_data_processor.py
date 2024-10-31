import os
import re
import gc
import numpy as np
import xarray as xr

def get_file_paths(base_dir, variable, model, start_year, end_year, suffix=""):
    # Update base directories based on the variable
    if variable == "rsds":
        base_dir = '/work/nai5/storage/interpolated/MPI/RSDS'
    elif variable == "uas":
        base_dir = '/work/nai5/storage/interpolated/MPI/UAS'
    elif variable == "vas":
        base_dir = '/work/nai5/storage/interpolated/MPI/VAS'
    elif variable == "tas":
        base_dir = '/datacommons/gpeos/nai5/var_data/Variables/tas/GFDL/interpolated'
    else:
        raise ValueError(f"Unknown variable: {variable}")

    # Return the file path based on the provided suffix
    if suffix:
        return f"{base_dir}/interpolated_{variable}_{model}_{start_year}-{end_year}_{suffix}.nc"
    else:
        return f"{base_dir}/interpolated_{variable}_{model}_{start_year}-{end_year}.nc"

def interpolate_model_to_target_grid(model_data, target_lats, target_lons):
    print("Starting interpolation...")
    primary_var = [var for var in model_data.data_vars if 'time' in model_data[var].dims and 'lat' in model_data[var].dims and 'lon' in model_data[var].dims][0]
    primary_data = model_data[primary_var]

    # Check for NaNs in the input data and fill them before interpolation
    if np.isnan(primary_data).any():
        print("Warning: NaNs found in input data. Filling NaNs with nearest values before interpolation.")
        primary_data = primary_data.ffill(dim='time').bfill(dim='time')
    
    print(f"Original data shape: {primary_data.shape}")

    # Perform interpolation using xarray's interp method
    interpolated_data = primary_data.interp(lat=target_lats, lon=target_lons, method="linear")

    print(f"Interpolated data shape: {interpolated_data.shape}")

    # Check for NaNs after interpolation
    nans_after_interp = np.isnan(interpolated_data).sum().item()
    total_values = interpolated_data.size
    nan_percentage = (nans_after_interp / total_values) * 100
    print(f"Number of NaNs after interpolation: {nans_after_interp}")
    print(f"Percentage of values that are NaN after interpolation: {nan_percentage:.2f}%")

    # Fill NaNs with the mean value of the data
    mean_value = interpolated_data.mean().item()
    interpolated_data = interpolated_data.fillna(mean_value)

    nans_after_filling = np.isnan(interpolated_data).sum().item()
    print(f"Number of NaNs after filling with mean value: {nans_after_filling}")

    return interpolated_data

def align_time_coordinates(reference_data, target_data):
    ref_time = reference_data.time
    tgt_time = target_data.time

    if ref_time.shape != tgt_time.shape or not np.array_equal(ref_time.values, tgt_time.values):
        target_data['time'] = ref_time
        print("Time coordinates aligned.")
    else:
        print("Time coordinates are already aligned.")
    return target_data

def count_nans(data, var_name):
    if isinstance(data, xr.DataArray):
        nans = np.isnan(data).sum().item()
    elif isinstance(data, xr.Dataset):
        nans = sum([np.isnan(data[v]).sum().item() for v in data.data_vars])
    else:
        raise TypeError("Input must be an xarray DataArray or Dataset.")
    print(f"Number of NaNs in {var_name}: {nans}")
    return nans

def save_and_overwrite(file_path, data, suffix, output_dir):
    # Modify the file name with the provided suffix
    new_file_name = re.sub(r'(\.nc)$', f'_{suffix}.nc', os.path.basename(file_path))
    
    # Construct the new file path in the specified output directory
    new_file_path = os.path.join(output_dir, new_file_name)
    
    # Save the new data directly to the output directory with the grid suffix
    data.to_netcdf(new_file_path)

def process_files(start_year, end_year, base_dir, output_dir):
    for year in range(start_year, end_year + 1, 2):
        start_year = year
        end_year = start_year + 1

        tas_file = get_file_paths(base_dir, "tas", "GFDL", start_year, end_year, "grid")
        rsds_file = get_file_paths(base_dir, "rsds", "MPI", start_year, end_year)
        uas_file = get_file_paths(base_dir, "uas", "MPI", start_year, end_year)
        vas_file = get_file_paths(base_dir, "vas", "MPI", start_year, end_year)

        print(f"Loading TAS dataset from file: {tas_file}")
        print(f"Loading RSDS dataset from file: {rsds_file}")
        print(f"Loading UAS dataset from file: {uas_file}")
        print(f"Loading VAS dataset from file: {vas_file}")

        tas_data = xr.open_dataset(tas_file).drop_vars('bnds', errors='ignore')
        rsds_data = xr.open_dataset(rsds_file).drop_vars('bnds', errors='ignore')
        uas_data = xr.open_dataset(uas_file).drop_vars('bnds', errors='ignore')
        vas_data = xr.open_dataset(vas_file).drop_vars('bnds', errors='ignore')

        print(f"TAS dataset shape: {tas_data['tas'].shape}")
        print(f"RSDS dataset shape: {rsds_data['rsds'].shape}")
        print(f"UAS dataset shape: {uas_data['uas'].shape}")
        print(f"VAS dataset shape: {vas_data['vas'].shape}")

        print("Slicing rsds, uas, and vas data to match tas time dimension...")

        time_length = len(tas_data.time)
        rsds_data_sliced = rsds_data.isel(time=slice(0, time_length))
        uas_data_sliced = uas_data.isel(time=slice(0, time_length))
        vas_data_sliced = vas_data.isel(time=slice(0, time_length))

        print("Sliced rsds data shape: ", rsds_data_sliced.rsds.shape)
        print("Sliced uas data shape: ", uas_data_sliced.uas.shape)
        print("Sliced vas data shape: ", vas_data_sliced.vas.shape)

        print("Checking NaNs in original datasets before interpolation...")
        count_nans(tas_data['tas'], 'tas')
        count_nans(rsds_data_sliced['rsds'], 'rsds')
        count_nans(uas_data_sliced['uas'], 'uas')
        count_nans(vas_data_sliced['vas'], 'vas')

        # Define target latitudes and longitudes from tas dataset
        target_lats = tas_data.lat.values
        target_lons = tas_data.lon.values

        print("Performing interpolation for rsds...")
        rsds_data_interp = interpolate_model_to_target_grid(rsds_data_sliced, target_lats, target_lons)

        print("Performing interpolation for uas...")
        uas_data_interp = interpolate_model_to_target_grid(uas_data_sliced, target_lats, target_lons)

        print("Performing interpolation for vas...")
        vas_data_interp = interpolate_model_to_target_grid(vas_data_sliced, target_lats, target_lons)

        tas_data_interp = tas_data.drop_vars(['lat_bnds', 'lon_bnds'], errors='ignore')
        rsds_data_interp = rsds_data_interp.drop_vars(['lat_bnds', 'lon_bnds'], errors='ignore')
        uas_data_interp = uas_data_interp.drop_vars(['lat_bnds', 'lon_bnds'], errors='ignore')
        vas_data_interp = vas_data_interp.drop_vars(['lat_bnds', 'lon_bnds'], errors='ignore')

        print("Aligning time coordinates...")
        rsds_data_interp = align_time_coordinates(tas_data, rsds_data_interp)
        uas_data_interp = align_time_coordinates(tas_data, uas_data_interp)
        vas_data_interp = align_time_coordinates(tas_data, vas_data_interp)

        print(f"Saving datasets for {start_year}-{end_year}...")

        save_and_overwrite(rsds_file, rsds_data_interp, 'grid', output_dir)
        save_and_overwrite(uas_file, uas_data_interp, 'grid', output_dir)
        save_and_overwrite(vas_file, vas_data_interp, 'grid', output_dir)

        tas_data.close()
        rsds_data.close()
        uas_data.close()
        vas_data.close()
        tas_data_interp.close()
        rsds_data_interp.close()
        uas_data_interp.close()
        vas_data_interp.close()

        gc.collect()

        print(f"Completed processing for {start_year}-{end_year}")

# Set the base directory and the years to process
base_dir = '/datacommons/gpeos/nai5/var_data'
start_year = 2019
end_year = 2100

output_dir = '/work/nai5/storage/interpolated/MPI/data'
process_files(start_year, end_year, base_dir, output_dir)
