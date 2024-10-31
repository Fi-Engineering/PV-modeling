import xarray as xr
import numpy as np
import pandas as pd
import pvlib
from scipy.interpolate import interp1d
import cftime
import gc

def load_datasets_for_case(case, year_start, year_end):
    # Adjust start year to align with 2-year chunks
    adjusted_start_year = year_start if year_start % 2 == 1 else year_start - 1
    adjusted_end_year = adjusted_start_year + 1

    # Load datasets for 2080-2100 period
    vas_file_path = f'./vas/CMCC/interpolated/interpolated_vas_CMCC_{adjusted_start_year}-{adjusted_end_year}.nc'
    uas_file_path = f'./uas/CMCC/interpolated/interpolated_uas_CMCC_{adjusted_start_year}-{adjusted_end_year}.nc'
    tas_file_path = f'./tas/GFDL/interpolated/interpolated_tas_GFDL_{adjusted_start_year}-{adjusted_end_year}_grid.nc'
    rsds_file_path = f'./rsds/CMCC/interpolated/interpolated_rsds_CMCC_{adjusted_start_year}-{adjusted_end_year}.nc'

    vas_ds = xr.open_dataset(vas_file_path)
    uas_ds = xr.open_dataset(uas_file_path)
    tas_ds = xr.open_dataset(tas_file_path)
    rsds_ds = xr.open_dataset(rsds_file_path)

    if case == 1:
        # Load RSDS dataset for 2020-2040 period
        adjusted_start_year_2020_2040 = adjusted_start_year - 60
        adjusted_end_year_2020_2040 = adjusted_end_year - 60
        rsds_file_path_2020_2040 = f'./rsds/CMCC/interpolated/interpolated_rsds_CMCC_{adjusted_start_year_2020_2040}-{adjusted_end_year_2020_2040}.nc'
        rsds_ds_2020_2040 = xr.open_dataset(rsds_file_path_2020_2040)

        # Print pre-change RSDS values
        print(f"Pre-change RSDS values for {adjusted_start_year}-{adjusted_end_year}:")
        print(rsds_ds['rsds'].values)

        # Replace RSDS values
        rsds_ds['rsds'] = (rsds_ds['rsds'].dims, rsds_ds_2020_2040['rsds'].values)
        
        # Print post-change RSDS values
        print(f"Post-change RSDS values for {adjusted_start_year}-{adjusted_end_year}:")
        print(rsds_ds['rsds'].values)
        
        print(f"Replaced RSDS values for {adjusted_start_year}-{adjusted_end_year} with values from {adjusted_start_year_2020_2040}-{adjusted_end_year_2020_2040}")

    elif case == 2:
        # Load TAS dataset for 2020-2040 period
        adjusted_start_year_2020_2040 = adjusted_start_year - 60
        adjusted_end_year_2020_2040 = adjusted_end_year - 60
        tas_file_path_2020_2040 = f'./tas/GFDL/interpolated/interpolated_tas_GFDL_{adjusted_start_year_2020_2040}-{adjusted_end_year_2020_2040}_grid.nc'
        tas_ds_2020_2040 = xr.open_dataset(tas_file_path_2020_2040)

        # Print pre-change TAS values
        print(f"Pre-change TAS values for {adjusted_start_year}-{adjusted_end_year}:")
        print(tas_ds['tas'].values)

        # Replace TAS values
        tas_ds['tas'] = (tas_ds['tas'].dims, tas_ds_2020_2040['tas'].values)
        
        # Print post-change TAS values
        print(f"Post-change TAS values for {adjusted_start_year}-{adjusted_end_year}:")
        print(tas_ds['tas'].values)
        
        print(f"Replaced TAS values for {adjusted_start_year}-{adjusted_end_year} with values from {adjusted_start_year_2020_2040}-{adjusted_end_year_2020_2040}")

    elif case == 3:
        # Load VAS and UAS datasets for 2020-2040 period
        adjusted_start_year_2020_2040 = adjusted_start_year - 60
        adjusted_end_year_2020_2040 = adjusted_end_year - 60
        vas_file_path_2020_2040 = f'./vas/CMCC/interpolated/interpolated_vas_CMCC_{adjusted_start_year_2020_2040}-{adjusted_end_year_2020_2040}.nc'
        uas_file_path_2020_2040 = f'./uas/CMCC/interpolated/interpolated_uas_CMCC_{adjusted_start_year_2020_2040}-{adjusted_end_year_2020_2040}.nc'
        vas_ds_2020_2040 = xr.open_dataset(vas_file_path_2020_2040)
        uas_ds_2020_2040 = xr.open_dataset(uas_file_path_2020_2040)

        # Print pre-change VAS and UAS values
        print(f"Pre-change VAS values for {adjusted_start_year}-{adjusted_end_year}:")
        print(vas_ds['vas'].values)
        print(f"Pre-change UAS values for {adjusted_start_year}-{adjusted_end_year}:")
        print(uas_ds['uas'].values)

        # Replace VAS and UAS values
        vas_ds['vas'] = (vas_ds['vas'].dims, vas_ds_2020_2040['vas'].values)
        uas_ds['uas'] = (uas_ds['uas'].dims, uas_ds_2020_2040['uas'].values)
        
        # Print post-change VAS and UAS values
        print(f"Post-change VAS values for {adjusted_start_year}-{adjusted_end_year}:")
        print(vas_ds['vas'].values)
        print(f"Post-change UAS values for {adjusted_start_year}-{adjusted_end_year}:")
        print(uas_ds['uas'].values)
        
        print(f"Replaced VAS and UAS values for {adjusted_start_year}-{adjusted_end_year} with values from {adjusted_start_year_2020_2040}-{adjusted_end_year_2020_2040}")

    aligned_apparent_zenith_ds = xr.open_dataset('apparent_zenith_two_years_2017_2018.nc')
    aligned_azimuth_ds = xr.open_dataset('azimuth_two_years_2017_2018.nc')
    print("Loaded aligned apparent zenith and azimuth datasets.")

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
    print("Calculated wind speed.")

    # Convert temperature from Kelvin to Celsius
    tas_ds['temperature'] = tas_ds['tas'] - 273.15
    print("Converted temperature from Kelvin to Celsius.")
    
    # Check for NaN values in the inputs
    if results_ds['poa_global'].isnull().any():
        print("Warning: 'poa_global' contains NaN values.")
    else:
        print("'poa_global' does not contain NaN values.")
        
    if tas_ds['temperature'].isnull().any():
        print("Warning: 'temperature' contains NaN values.")
    else:
        print("'temperature' does not contain NaN values.")
        
    if uas_ds['wind_speed'].isnull().any():
        print("Warning: 'wind_speed' contains NaN values.")
    else:
        print("'wind_speed' does not contain NaN values.")
    
    # Define temperature model parameters
    temp_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
    temp_params['deltaT'] = float(temp_params['deltaT'])
    print("Temperature model parameters defined.")
    
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
        nan_locs = tas_ds['temperature_cell'].isnull()
        print("Logging conditions that lead to NaN values in temperature_cell:")
        print(f"poa_global at NaN locations: {results_ds['poa_global'].where(nan_locs, drop=True).values}")
        print(f"temperature at NaN locations: {tas_ds['temperature'].where(nan_locs, drop=True).values}")
        print(f"wind_speed at NaN locations: {uas_ds['wind_speed'].where(nan_locs, drop=True).values}")
    else:
        print("'temperature_cell' does not contain NaN values after calculation.")
    
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
    print(f"Initial effective_irradiance dimensions: {effective_irradiance.shape}")
    print(f"Initial temperature_cell dimensions: {temperature_cell.shape}")
    
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
            #print(f"\nEffective irradiance at time step {time_step} shape: {effective_irradiance_step.shape}")
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
        print(f"SAPM result {key} shape after stacking: {sapm_data.shape}")
        sapm_results[key] = xr.DataArray(sapm_data, dims=['time', 'lat', 'lon'], coords={'time': effective_irradiance.time, 'lat': effective_irradiance.lat, 'lon': effective_irradiance.lon})
    
    print(f"\nSAPM results shapes:")
    for key in sapm_results:
        print(f"{key} shape: {sapm_results[key].shape}")

    return sapm_results



def calculate_ac_power(v_mp, p_mp, inverter_properties):
    #print(f"Shape of v_mp: {v_mp.shape}")
    #print(f"Shape of p_mp: {p_mp.shape}")
    
    ac_power_array = pvlib.inverter.sandia(v_mp, p_mp, inverter_properties)
    print(f"Shape of ac_power_array: {ac_power_array.shape}")
    
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


def main():
    for case in range(1, 4):
        for year in range(2093, 2100, 2):
            year_start = year
            year_end = year + 1
            adjusted_start_year = year_start if year_start % 2 == 1 else year_start - 1
            adjusted_end_year = adjusted_start_year + 1
            
            output_file_path = f'output/daily_energy_output_{adjusted_start_year}-{adjusted_end_year}_case_{case}.nc'
            
            vas_ds, uas_ds, tas_ds, rsds_ds, aligned_apparent_zenith_ds, aligned_azimuth_ds = load_datasets_for_case(case, year_start, year_end)
            
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
        
            # Preprocess temperature and wind speed data
            uas_ds['wind_speed'] = np.sqrt(uas_ds['uas']**2 + vas_ds['vas']**2)
            tas_ds['temperature'] = tas_ds['tas'] - 273.15
        
            # Check intermediate results before calculating temperature_cell
            print(f"Intermediate results before temperature_cell calculation for case {case}, year {year_start}-{year_end}:")
            print(f"poa_global min: {results_ds['poa_global'].min().values}, max: {results_ds['poa_global'].max().values}")
            print(f"temperature min: {tas_ds['temperature'].min().values}, max: {tas_ds['temperature'].max().values}")
            print(f"wind_speed min: {uas_ds['wind_speed'].min().values}, max: {uas_ds['wind_speed'].max().values}")
        
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
        
            for key in sapm_results:
                sapm_data = sapm_results[key]
                print(f"SAPM result {key} shape before transposing: {sapm_data.shape}")
        
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
        
            daily_energy_ds.to_netcdf(output_file_path)
            
            print(f"Results saved successfully for {adjusted_start_year}-{adjusted_end_year} case {case}.")
            
            vas_ds.close()
            uas_ds.close()
            tas_ds.close()
            rsds_ds.close()
            aligned_apparent_zenith_ds.close()
            aligned_azimuth_ds.close()
            del vas_ds, uas_ds, tas_ds, rsds_ds, aligned_apparent_zenith_ds, aligned_azimuth_ds
            del results_ds
            gc.collect()

if __name__ == "__main__":
    main()
