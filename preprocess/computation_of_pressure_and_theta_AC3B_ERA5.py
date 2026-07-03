# Computation of pressure from Surface pressure and of potential temperature of ERA5 forecasts in model levels

import numpy as np
import xarray as xr

import pandas as pd
from matplotlib import pyplot as plt
import os



local = False # Set to True if running on local lmd machine, False if running on spirit


lon_min = -180
lon_max = 180

lat_min = 74
lat_max = 90

root_input = "/data/nchiab/PV"

if local:
    root_input = "/media/chabranoo/LaCie/PhD"


data_folder= "/data/nchiab/PV/Data/era5_forecasts/"

#path_t = f"{root_input}/Data/era5_forecasts/ERA5_TEMP_all_levels_2022-08-16_50N-90N.nc"
path_t1 = f"{data_folder}/ERA5_TEMP_all_levels_2022-08-17_18:00:00_50N-90N.nc"
path_t2 = f"{data_folder}/ERA5_TEMP_all_levels_2022-08-18_06:00:00_50N-90N.nc"
path_t3 = f"{data_folder}/ERA5_TEMP_all_levels_2022-08-18_18:00:00_50N-90N.nc"
path_t4 = f"{data_folder}/ERA5_TEMP_all_levels_2022-08-19_06:00:00_50N-90N.nc"
#path_p = f"{root_input}/Data/era5_forecasts/ERA5_pres_all_levels_2022-08-16_50N-90N.nc"

#path_sp = f"{root_input}/Data/era5_forecasts/ERA5_SP_2022-08-16_50N-90N.nc"

path_sp1 = f"{data_folder}/ERA5_SP_2022-08-17_18:00:00_50N-90N.nc"
path_sp2 = f"{data_folder}/ERA5_SP_2022-08-18_06:00:00_50N-90N.nc"
path_sp3 = f"{data_folder}/ERA5_SP_2022-08-18_18:00:00_50N-90N.nc"
path_sp4 = f"{data_folder}/ERA5_SP_2022-08-19_06:00:00_50N-90N.nc"

t1 = xr.open_dataset(path_t1)['t']
t2 = xr.open_dataset(path_t2)['t']
t3= xr.open_dataset(path_t3)['t']
t4 = xr.open_dataset(path_t4)['t']

t = xr.concat([t1, t2, t3, t4], dim='valid_time')

sp1 = xr.open_dataset(path_sp1)['sp']
sp2 = xr.open_dataset(path_sp2)['sp']
sp3 = xr.open_dataset(path_sp3)['sp']
sp4 = xr.open_dataset(path_sp4)['sp']

sp = xr.concat([sp1, sp2, sp3, sp4], dim='valid_time')



mod_lev_txt = "../Niveaux_modeles_IFS_137.txt"  # Replace with your actual file path
df = pd.read_csv(mod_lev_txt, sep=r'\s+', header=None, skiprows=1)

df.columns = ["nk", "ak", "bk", "phk", "pfk", "geopaltk", "zaltk", "tempk", "rhok"]

df.replace('-', pd.NA, inplace=True)

#print(df)

sp_extand = sp.expand_dims({'model_level': t['model_level']})

ak = df['ak'].to_numpy().reshape(137, 1, 1, 1)
bk = df['bk'].to_numpy().reshape(137, 1, 1, 1)

#p = xr.open_dataset(path_p)['pres']

#compute p = ak + bk * sp

p_comp =  ak + bk * sp_extand

p_comp.name = 'pres'
p_comp = p_comp.assign_attrs(long_name="Pressure", standard_name = "air_pressure").transpose('valid_time', 'model_level', 'latitude', 'longitude')



def theta_f(T, P, P0=1000):
    return T * (P0 / P)**0.286







theta = theta_f(t, p_comp, P0=100000)


sp = sp.rename({'valid_time': 'time'}).sel(latitude = slice(lat_max, lat_min), longitude = slice(lon_min, lon_max) )


theta.name = 'theta'
theta = theta.assign_attrs(long_name="Potential temperature", standard_name = "air_potential_temperature")

#theta.to_netcdf(f"{root_input}/Data/era5_forecasts/ERA5_theta_all_levels_2022-08-17a19_50N-90N.nc")
#p_comp.to_netcdf(f"{root_input}/Data/era5_forecasts/ERA5_pres_all_levels_2022-08-17a19_50N-90N.nc")


sp.to_netcdf(f"{root_input}/Data/era5_forecasts/ERA5_SP_2022-08-17a19_{lat_min}N-{lat_max}N.nc")
