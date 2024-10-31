import xarray as xr
import numpy as np
import pandas as pd
import pvlib
from scipy.interpolate import interp1d
import cftime
import gc


def load_datasets(year_start, year_end):
    vas_file_path = f'/work/nai5/storage/interpolated/MPI/data/interpolated_vas_MPI_{year_start}-{year_end}_grid.nc'
    uas_file_path = f'/work/nai5/storage/interpolated/MPI/data/interpolated_uas_MPI_{year_start}-{year_end}_grid.nc'
    tas_file_path = f'/datacommons/gpeos/nai5/var_data/Variables/tas/GFDL/interpolated/interpolated_tas_GFDL_{year_start}-{year_end}_grid.nc'
    rsds_file_path = f'/work/nai5/storage/interpolated/MPI/data/interpolated_rsds_MPI_{year_start}-{year_end}_grid.nc'
    
    # Load datasets here (assuming using xarray)
    vas_ds = xr.open_dataset(vas_file_path)
    uas_ds = xr.open_dataset(uas_file_path)
    tas_ds = xr.open_dataset(tas_file_path)
    rsds_ds = xr.open_dataset(rsds_file_path)

    aligned_apparent_zenith_ds = xr.open_dataset('apparent_zenith_two_years_2017_2018.nc')
    aligned_azimuth_ds = xr.open_dataset('azimuth_two_years_2017_2018.nc')
    
    print(f"Datasets for {year_start}-{year_end} loaded successfully.")

    return vas_ds, uas_ds, tas_ds, rsds_ds, aligned_apparent_zenith_ds, aligned_azimuth_ds

def load_pvlib_data():
    sandia_module = pvlib.pvsystem.retrieve_sam(name='SandiaMod')
    sapm_inverter = pvlib.pvsystem.retrieve_sam(name='CECInverter')
    
    module_name = 'Schott_Solar_ASE_300_DGF_50__320___2007__E__'
    module_properties = sandia_module[module_name]
    
    inverter_name = 'ABB__PVI_3_0_OUTD_S_US_Z__240V_'
    inverter_properties = sapm_inverter[inverter_name]
    
    print("Module and inverter loaded successfully.")
    return module_properties, inverter_properties

def prepare_surface_data(rsds_ds):
    lat_2d, lon_2d = np.meshgrid(rsds_ds['lat'], rsds_ds['lon'], indexing='ij')
    surface_tilt_2d = np.where(np.abs(lat_2d) > 40, 40, np.abs(lat_2d))
    surface_azimuth_2d = 180
    
    surface_tilt = xr.DataArray(surface_tilt_2d, coords={'lat': rsds_ds['lat'], 'lon': rsds_ds['lon']})
    surface_azimuth = xr.DataArray(surface_azimuth_2d, coords={'lat': rsds_ds['lat'], 'lon': rsds_ds['lon']}, dims=['lat', 'lon'])
    
    print("Surface tilt and surface azimuth loaded successfully.")
    return surface_tilt, surface_azimuth

def replace_time_dimension(ds_to_modify, target_time):
    """
    Replaces the time dimension of the input dataset to match the target time dimension.
    
    Parameters:
    ds_to_modify (xr.Dataset): The dataset to modify.
    target_time (xr.DataArray): The target time dimension to replace with.
    
    Returns:
    xr.Dataset: The modified dataset with the new time dimension.
    """
    ds_to_modify = ds_to_modify.copy()
    ds_to_modify = ds_to_modify.assign_coords(time=target_time)
    return ds_to_modify

def calculate_dni_extra(rsds_ds):
    day_of_year = rsds_ds.time.dt.dayofyear
    dni_extra = pvlib.irradiance.get_extra_radiation(day_of_year)
    print("DNI Extra loaded successfully.")
    return dni_extra

def calculate_airmass(apparent_zenith):
    mask_zenith_above_90 = apparent_zenith >= 90
    valid_apparent_zenith = np.where(mask_zenith_above_90, np.nan, apparent_zenith.values)
    airmass_values = pvlib.atmosphere.get_relative_airmass(valid_apparent_zenith)
    airmass = xr.DataArray(airmass_values, coords=apparent_zenith.coords, dims=apparent_zenith.dims)
    print("Airmass loaded successfully.")
    return airmass

def calculate_dni_all(rsds_ds, aligned_apparent_zenith_ds):
    if hasattr(rsds_ds.time.values[0], 'strftime'):
        rsds_ds['time'] = pd.to_datetime([dt.strftime('%Y-%m-%d %H:%M:%S') for dt in rsds_ds.time.values])
    
    MIN_COS_ZENITH_VALUE = 0.03  # Corresponds to a zenith angle slightly more than 88 degrees

    # Adjust the cosine of the zenith angle to not go below the MIN_COS_ZENITH_VALUE
    cos_zenith_adjusted = np.maximum(np.cos(np.deg2rad(aligned_apparent_zenith_ds['apparent_zenith'])), MIN_COS_ZENITH_VALUE)

    # Calculate DNI using the adjusted cosine zenith angle, without needing a separate conditions_met
    dni_all_adjusted = rsds_ds['rsds'] * cos_zenith_adjusted
    dni_all_adjusted = xr.where(dni_all_adjusted < 0, 0, dni_all_adjusted)  # Ensuring DNI is not negative

    print(f"cos_zenith_adjusted shape: {cos_zenith_adjusted.shape}")
    print(f"rsds_ds['rsds'] shape: {rsds_ds['rsds'].shape}")
    print(f"rsds_ds time dimension: {rsds_ds['time'].shape}")
    print(f"cos_zenith_adjusted time dimension: {aligned_apparent_zenith_ds['time'].shape}")
    print(f"dni_all_adjusted initial shape: {dni_all_adjusted.shape}")

    # Create a new DataArray with the correct coordinates and dimensions order (lat, lon, time)
    coords = {'lat': rsds_ds['lat'], 'lon': rsds_ds['lon'], 'time': rsds_ds['time']}
    dni_all_adjusted = xr.DataArray(dni_all_adjusted.values, coords=coords, dims=['lat', 'lon', 'time'])

    print(f"dni_all_adjusted shape after reassignment: {dni_all_adjusted.shape}")
    print(f"dni_all_adjusted coords: {dni_all_adjusted.coords}")

    # Initialize results_ds with the correct dimensions and coordinates
    results_ds = xr.Dataset(coords={
        'time': rsds_ds['time'],
        'lat': rsds_ds['lat'],
        'lon': rsds_ds['lon']
    })

    results_ds['dni_all'] = dni_all_adjusted

    print("DNI_All loaded successfully")
    return results_ds

def calculate_aoi(surface_tilt, surface_azimuth, apparent_zenith, azimuth):
    aoi = pvlib.irradiance.aoi(surface_tilt, surface_azimuth, apparent_zenith, azimuth)
    aoi = aoi.transpose('time', 'lat', 'lon')
    print("AOI data loaded successfully.")
    return aoi

def calculate_poa_components(surface_tilt, surface_azimuth, rsds, dni_all, dni_extra, apparent_zenith, azimuth):
    poa_sky_diffuse_all = None
    try:
        poa_sky_diffuse_all = pvlib.irradiance.haydavies(
            surface_tilt,
            surface_azimuth,
            rsds,
            dni_all,
            dni_extra.broadcast_like(dni_all),
            apparent_zenith,
            azimuth
        )
    except Exception as e:
        print("An error occurred during computation:", str(e))
    
    poa_ground_diffuse_all = pvlib.irradiance.get_ground_diffuse(surface_tilt, rsds, albedo=0.2)
    print("Hay Davies and ground diffuse irradiances data loaded successfully.")
    return poa_sky_diffuse_all, poa_ground_diffuse_all

def calculate_temperature_cell(results_ds, tas_ds, uas_ds, vas_ds):
    # Calculate wind speed
    uas_ds['wind_speed'] = np.sqrt(uas_ds['uas']**2 + vas_ds['vas']**2)
    
    # Convert temperature from Kelvin to Celsius
    tas_ds['temperature'] = tas_ds['tas'] - 273.15
    
    # Check for NaN values in the inputs
    if results_ds['poa_global'].isnull().any():
        print("Warning: 'poa_global' contains NaN values.")
    if tas_ds['temperature'].isnull().any():
        print("Warning: 'temperature' contains NaN values.")
    if uas_ds['wind_speed'].isnull().any():
        print("Warning: 'wind_speed' contains NaN values.")
    
    # Define temperature model parameters
    temp_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
    temp_params['deltaT'] = float(temp_params['deltaT'])
    
    # Calculate cell temperature
    tas_ds['temperature_cell'] = pvlib.temperature.sapm_cell(
        results_ds['poa_global'],
        tas_ds['temperature'],
        uas_ds['wind_speed'],
        **temp_params
    )
    
    # Check for NaN values in the output
    if tas_ds['temperature_cell'].isnull().any():
        print("Warning: 'temperature_cell' contains NaN values after calculation.")
    
    print("Cell temperature data loaded successfully.")
    return tas_ds['temperature_cell']

def calculate_effective_irradiance(poa_direct, poa_diffuse, airmass, aoi, module_properties):
    # Ensure all arrays have the same dimension order (lat, lon, time)
    poa_direct = poa_direct.transpose('lat', 'lon', 'time')
    poa_diffuse = poa_diffuse.transpose('lat', 'lon', 'time')
    airmass = airmass.transpose('lat', 'lon', 'time')
    aoi = aoi.transpose('lat', 'lon', 'time')

    effective_irradiance = pvlib.pvsystem.sapm_effective_irradiance(
        poa_direct,
        poa_diffuse,
        airmass,
        aoi,
        module_properties
    )
    print("Effective Irradiance data loaded successfully.")
    return effective_irradiance

def print_dataset_info(ds, name):
    print(f"\n{name} Dataset Info:")
    print(f"Dimensions: {ds.dims}")
    print(f"Coordinates: {ds.coords}")
    for var in ds.data_vars:
        print(f"{var} shape: {ds[var].shape}")
        
def calculate_sapm(effective_irradiance, temperature_cell, module_properties):
    # Print initial dimensions of inputs
    print(f"Effective Irradiance dimensions: {effective_irradiance.shape}")
    print(f"Temperature Cell dimensions: {temperature_cell.shape}")
    
    # Calculate total number of elements
    total_elements = np.prod(effective_irradiance.shape)
    
    # Print min, max, and NaN counts for inputs
    effective_irradiance_nans = np.isnan(effective_irradiance).sum().values
    temperature_cell_nans = np.isnan(temperature_cell).sum().values

    print(f"Effective Irradiance - Min: {effective_irradiance.min().values}, Max: {effective_irradiance.max().values}, NaNs: {effective_irradiance_nans} ({(effective_irradiance_nans / total_elements) * 100:.2f}%)")
    print(f"Temperature Cell - Min: {temperature_cell.min().values}, Max: {temperature_cell.max().values}, NaNs: {temperature_cell_nans} ({(temperature_cell_nans / total_elements) * 100:.2f}%)")
    
    # Extract the number of time steps
    n_time_steps = effective_irradiance.shape[2]
    print(f"Number of time steps: {n_time_steps}")

    sapm_results = {key: [] for key in ['i_sc', 'i_mp', 'v_oc', 'v_mp', 'p_mp', 'i_x', 'i_xx']}
    
    for time_step in range(n_time_steps):
        if (time_step + 1) % 100 == 0 or time_step == n_time_steps - 1:
            print(f"Processing time step {time_step + 1}/{n_time_steps}", end='\r')

        # Extract single time step data for effective_irradiance and temperature_cell
        effective_irradiance_step = effective_irradiance.isel(time=time_step)
        temperature_cell_step = temperature_cell.isel(time=time_step)

        # Debugging prints for shapes
        if time_step < 5:  # Print details for the first few time steps
            print(f"\nEffective irradiance at time step {time_step} shape: {effective_irradiance_step.shape}")
            print(f"Temperature cell at time step {time_step} shape: {temperature_cell_step.shape}")

        sapm_output = pvlib.pvsystem.sapm(
            effective_irradiance_step,
            temperature_cell_step,
            module_properties
        )

        for key in sapm_output:
            sapm_results[key].append(sapm_output[key])

    for key in sapm_results:
        sapm_data = np.stack(sapm_results[key], axis=0)  # Stack along the time dimension
        sapm_results[key] = xr.DataArray(sapm_data, dims=['time', 'lat', 'lon'], coords={'time': effective_irradiance.time, 'lat': effective_irradiance.lat, 'lon': effective_irradiance.lon})
    
    # Check for NaNs in v_mp and identify problematic indices
    v_mp = sapm_results['v_mp']
    v_mp_nans = np.isnan(v_mp).sum()
    print(f"v_mp NaNs: {v_mp_nans} ({(v_mp_nans / total_elements) * 100:.2f}%)")
    problematic_indices = np.where(np.isnan(v_mp))
    print(f"Problematic indices in v_mp: {problematic_indices}")
    
    # Check for out-of-bounds indices
    time_idx, lat_idx, lon_idx = problematic_indices
    
    out_of_bounds_lat = lat_idx >= effective_irradiance.shape[0]
    out_of_bounds_lon = lon_idx >= effective_irradiance.shape[1]
    out_of_bounds_time = time_idx >= effective_irradiance.shape[2]

    if np.any(out_of_bounds_lat):
        print(f"Out-of-bounds latitude indices: {lat_idx[out_of_bounds_lat]}")
    if np.any(out_of_bounds_lon):
        print(f"Out-of-bounds longitude indices: {lon_idx[out_of_bounds_lon]}")
    if np.any(out_of_bounds_time):
        print(f"Out-of-bounds time indices: {time_idx[out_of_bounds_time]}")
    
    # Only access elements when indices are valid
    valid_indices = ~out_of_bounds_lat & ~out_of_bounds_lon & ~out_of_bounds_time
    
    if np.any(valid_indices):
        valid_lat_idx = lat_idx[valid_indices]
        valid_lon_idx = lon_idx[valid_indices]
        valid_time_idx = time_idx[valid_indices]

        print(f"Valid problematic Effective Irradiance values: {effective_irradiance.values[valid_lat_idx, valid_lon_idx, valid_time_idx]}")
        print(f"Valid problematic Temperature Cell values: {temperature_cell.values[valid_lat_idx, valid_lon_idx, valid_time_idx]}")

    print(f"\nSAPM results shapes:")
    for key in sapm_results:
        print(f"{key} shape: {sapm_results[key].shape}")

    return sapm_results

def calculate_ac_power(v_mp, p_mp, inverter_properties):
    # Calculate total number of elements
    total_elements = np.prod(v_mp.shape)
    
    # Print min, max, and NaN counts for v_mp and p_mp
    v_mp_nans = np.isnan(v_mp).sum()
    p_mp_nans = np.isnan(p_mp).sum()
    print(f"v_mp - Min: {v_mp.min()}, Max: {v_mp.max()}, NaNs: {v_mp_nans} ({(v_mp_nans / total_elements) * 100:.2f}%)")
    print(f"p_mp - Min: {p_mp.min()}, Max: {p_mp.max()}, NaNs: {p_mp_nans} ({(p_mp_nans / total_elements) * 100:.2f}%)")
    
    ac_power_array = pvlib.inverter.sandia(v_mp, p_mp, inverter_properties)
    
    ac_power_nans = np.isnan(ac_power_array).sum()
    print(f"Shape of ac_power_array: {ac_power_array.shape}")
    print(f"AC Power - Min: {np.nanmin(ac_power_array)}, Max: {np.nanmax(ac_power_array)}, NaNs: {ac_power_nans} ({(ac_power_nans / total_elements) * 100:.2f}%)")
    print(f"\nAC power data loaded successfully.")
    return ac_power_array


def calculate_daily_energy(ac_power_dataarray):
    days = ac_power_dataarray['time'].dt.floor('D')
    grouped = ac_power_dataarray.groupby(days).mean(dim='time')
    daily_energy_Wh = grouped * 24
    print(f"\nDaily Energy data loaded successfully.")
    return daily_energy_Wh
'''
def apply_land_sea_mask(daily_energy_ds, land_sea_mask_ds):
    land_sea_mask = land_sea_mask_ds['mask']
    land_sea_mask_reindexed = land_sea_mask.reindex_like(daily_energy_ds, method='nearest')
    
    # Transpose daily_energy_ds to match the mask dimensions
    daily_energy_transposed = daily_energy_ds['daily_energy'].transpose('lat', 'lon', 'time')

    # Expand land-sea mask to match the time dimension of daily_energy_ds
    land_sea_mask_expanded = land_sea_mask_reindexed.expand_dims(time=daily_energy_ds['time']).transpose('lat', 'lon', 'time')

    # Apply the land-sea mask to the daily energy dataset
    masked_daily_energy = xr.where(land_sea_mask_expanded == 1, daily_energy_transposed, np.nan)
    
    masked_daily_energy_ds = daily_energy_ds.copy()
    masked_daily_energy_ds['daily_energy'] = masked_daily_energy
    
    print(f"\nTime renamed and vars dropped successfully.")
    return masked_daily_energy_ds
'''
def check_for_nan(data_array, name):
    if data_array.isnull().any():
        print(f"Warning: {name} contains NaN values.")

def cftime_to_datetimeindex(time_var):
    # Check if the time variable is of type cftime or numpy.datetime64
    if isinstance(time_var.values[0], cftime.DatetimeNoLeap):
        return pd.to_datetime([pd.Timestamp(ct.year, ct.month, ct.day, ct.hour, ct.minute, ct.second) 
                               for ct in time_var.values])
    else:
        return pd.to_datetime(time_var.values)
'''
# Function to check if time points are uniformly spaced
def check_uniform_spacing(time_array):
    time_diffs = np.diff(time_array).astype('timedelta64[s]').astype('float64')
    uniform = np.allclose(time_diffs, time_diffs[0])
    return uniform
# Function to interpolate datasets to match the target time points
def interpolate_to_target_time(ds, target_time):
    interp_func = interp1d(ds['time'].astype(float), ds.values, axis=0, fill_value='extrapolate')
    interpolated_values = interp_func(target_time.astype(float))
    interpolated_ds = xr.DataArray(interpolated_values, coords={'time': target_time, 'lat': ds['lat'], 'lon': ds['lon']}, dims=['time', 'lat', 'lon'])
    return interpolated_ds
'''

def main():
    for year in range(2019, 2100, 2):
        year_start = year
        year_end = year + 1
        output_file_path = f'/work/nai5/storage/MPI_output/MPI_daily_energy_output_{year_start}-{year_end}.nc'
        
        vas_ds, uas_ds, tas_ds, rsds_ds, aligned_apparent_zenith_ds, aligned_azimuth_ds = load_datasets(year_start, year_end)
        
        # Convert cftime to datetime using a custom function
        for ds in [vas_ds, uas_ds, tas_ds, rsds_ds, aligned_apparent_zenith_ds, aligned_azimuth_ds]:
            ds['time'] = cftime_to_datetimeindex(ds['time'])
    
        def drop_bnds(ds):
            if 'bnds' in ds.dims:
                ds = ds.drop_dims('bnds')
            return ds
    
        datasets = [vas_ds, uas_ds, tas_ds, rsds_ds, aligned_apparent_zenith_ds, aligned_azimuth_ds]
        for i, ds in enumerate(datasets):
            ds = drop_bnds(ds)
            if set(ds.dims) >= {'lat', 'lon', 'time'}:
                datasets[i] = ds.transpose('lat', 'lon', 'time')
            print(f"Shape of dataset {i} after dropping bnds and transposing: {datasets[i].dims}")
    
        vas_ds, uas_ds, tas_ds, rsds_ds, aligned_apparent_zenith_ds, aligned_azimuth_ds = datasets
        
        print(f"RSDS - NaN Count: {np.isnan(rsds_ds['rsds']).sum().values}")
        print(f"TAS - NaN Count: {np.isnan(tas_ds['tas']).sum().values}")
        print(f"UAS - NaN Count: {np.isnan(uas_ds['uas']).sum().values}")
        print(f"VAS - NaN Count: {np.isnan(vas_ds['vas']).sum().values}")

        vas_time = vas_ds['time'].values
        uas_time = uas_ds['time'].values
        tas_time = tas_ds['time'].values
        rsds_time = rsds_ds['time'].values
        apparent_zenith_time = aligned_apparent_zenith_ds['time'].values
        azimuth_time = aligned_azimuth_ds['time'].values
    
        common_time = rsds_time
        if len(vas_time) == len(uas_time) == len(tas_time) == len(rsds_time):
            print("All four datasets have the same number of time points.")
        else:
            raise ValueError("The time points of VAS, UAS, TAS, and RSDS datasets do not match.")
    
        aligned_apparent_zenith_ds = replace_time_dimension(aligned_apparent_zenith_ds, rsds_ds['time'])
        aligned_azimuth_ds = replace_time_dimension(aligned_azimuth_ds, rsds_ds['time'])
        
        print("Replaced time dimension for apparent_zenith and azimuth to match the common time points.")
        
        apparent_zenith = aligned_apparent_zenith_ds['apparent_zenith']
        azimuth = aligned_azimuth_ds['azimuth']
        
        print("Apparent_zenith shape:", apparent_zenith.shape)
        print("Azimuth shape:", azimuth.shape)
    
        module_properties, inverter_properties = load_pvlib_data()
        surface_tilt, surface_azimuth = prepare_surface_data(rsds_ds)
        results_ds = calculate_dni_all(rsds_ds, aligned_apparent_zenith_ds)
        
        dni_extra = calculate_dni_extra(rsds_ds)
        results_ds['dni_extra'] = dni_extra
    
        airmass = calculate_airmass(apparent_zenith)
        results_ds['airmass'] = airmass
        print("Airmass shape:", airmass.shape)
        check_for_nan(results_ds['dni_all'], 'DNI_All')
        
        aoi = calculate_aoi(surface_tilt, surface_azimuth, apparent_zenith, azimuth)
        results_ds['aoi'] = aoi
        check_for_nan(results_ds['aoi'], 'AOI')
    
        poa_sky_diffuse_all, poa_ground_diffuse_all = calculate_poa_components(surface_tilt, surface_azimuth, rsds_ds['rsds'], results_ds['dni_all'], dni_extra, apparent_zenith, azimuth)
        if poa_sky_diffuse_all is not None:
            results_ds['poa_sky_diffuse'] = poa_sky_diffuse_all
            check_for_nan(results_ds['poa_sky_diffuse'], 'POA_Sky_Diffuse')
        results_ds['poa_ground_diffuse'] = poa_ground_diffuse_all
        check_for_nan(results_ds['poa_ground_diffuse'], 'POA_Ground_Diffuse')
        
        poa_direct = results_ds['dni_all'] * np.cos(np.deg2rad(results_ds['aoi']))
        poa_diffuse = results_ds['poa_sky_diffuse'] + results_ds['poa_ground_diffuse']
        results_ds['poa_direct'] = poa_direct.transpose('lat', 'lon', 'time')
        results_ds['poa_diffuse'] = poa_diffuse.transpose('lat', 'lon', 'time')
        results_ds['poa_global'] = poa_direct + poa_diffuse
        check_for_nan(results_ds['poa_global'], 'POA_Global')
    
        tas_ds['temperature_cell'] = calculate_temperature_cell(results_ds, tas_ds, uas_ds, vas_ds)
        results_ds['temperature_cell'] = tas_ds['temperature_cell']
        check_for_nan(results_ds['temperature_cell'], 'Temperature_Cell')
    
        effective_irradiance = calculate_effective_irradiance(results_ds['poa_direct'], results_ds['poa_diffuse'], results_ds['airmass'], results_ds['aoi'], module_properties)
        results_ds['effective_irradiance'] = effective_irradiance
        check_for_nan(results_ds['effective_irradiance'], 'Effective_Irradiance')
    
        sapm_results = calculate_sapm(results_ds['effective_irradiance'], tas_ds['temperature_cell'], module_properties)
    
        print("\nResults_ds shapes before assignment:")
        for var in results_ds:
            print(f"{var} shape: {results_ds[var].shape}")
    
        # Check dimensions before assignment
        for key in sapm_results:
            sapm_data = sapm_results[key]
            print(f"SAPM result {key} shape before transposing: {sapm_data.shape}")
    
            # Ensure sapm_data has the correct dimensions
            if sapm_data.dims != ('time', 'lat', 'lon'):
                raise ValueError(f"SAPM result {key} has unexpected dimensions: {sapm_data.dims}")
    
            results_ds[key] = sapm_data
            check_for_nan(results_ds[key], key)
    
        print("\nResults_ds shapes after assignment:")
        for var in results_ds:
            print(f"{var} shape: {results_ds[var].shape}")
    
        ac_power_array = calculate_ac_power(results_ds['v_mp'].values, results_ds['p_mp'].values, inverter_properties)
        
        ac_power_array = np.transpose(ac_power_array, (1, 2, 0))
        
        ac_power_dataarray = xr.DataArray(ac_power_array, dims=('lat', 'lon', 'time'), coords={'time': results_ds['time'], 'lat': results_ds['lat'], 'lon': results_ds['lon']})
        check_for_nan(ac_power_dataarray, 'AC_Power')
        
        daily_energy_Wh = calculate_daily_energy(ac_power_dataarray)
        daily_energy_ds = daily_energy_Wh.to_dataset(name='daily_energy')
        daily_energy_ds = daily_energy_ds.rename({'floor': 'time'})
        check_for_nan(daily_energy_ds['daily_energy'], 'Daily_Energy')
    
        # land_sea_mask_ds = land_sea_mask.drop_vars(['lat_bnds', 'lon_bnds'])
        # masked_daily_energy_ds = apply_land_sea_mask(daily_energy_ds, land_sea_mask_ds)
    
        # masked_daily_energy_ds.to_netcdf(output_file_path)
        daily_energy_ds.to_netcdf(output_file_path)
        
        print(f"Results saved successfully for {year_start}-{year_end}.")
        
        # Close datasets and run garbage collection
        vas_ds.close()
        uas_ds.close()
        tas_ds.close()
        rsds_ds.close()
        aligned_apparent_zenith_ds.close()
        aligned_azimuth_ds.close()
        # land_sea_mask.close()
        del vas_ds, uas_ds, tas_ds, rsds_ds, aligned_apparent_zenith_ds, aligned_azimuth_ds #, land_sea_mask
        del results_ds
        gc.collect()
        

if __name__ == "__main__":
    main()
