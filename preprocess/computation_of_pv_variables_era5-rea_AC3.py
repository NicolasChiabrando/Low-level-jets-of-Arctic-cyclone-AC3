import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import csv
import pandas as pd
import dask
import cftime
import sys
import os

# %%
dataset = 'ERA5'  # 'era5'
day = "15a19" #"15a16" # "15a16" #"22" #
lon_min = -30
lon_max = 70

lat_min = 65
lat_max = 86



local = False
chunks = {'valid_time': 3} # 'model_level': 1
root_input = "/data/nchiab/PV"

if local:
    root_input = "/media/chabranoo/LaCie/PhD"




path_u = f"{root_input}/Data/{dataset}/ERA5_reanalyse_U_all_levels_{day}_50N-90N.nc"
path_v = f"{root_input}/Data/{dataset}/ERA5_reanalyse_V_all_levels_{day}_50N-90N.nc"
path_w = f"{root_input}/Data/{dataset}/ERA5_reanalyse_Omega_all_levels_{day}_50N-90N.nc"
path_t = f"{root_input}/Data/{dataset}/ERA5_reanalyse_TEMP_all_levels_{day}_50N-90N.nc"
path_p = f"{root_input}/Data/{dataset}/ERA5_reanalyse_pres_all_levels_{day}_50N-90N.nc"
path_sp = f"{root_input}/Data/{dataset}/ERA5_reanalyse_SP_{day}_50N-90N.nc"
path_theta = f"{root_input}/Data/{dataset}/ERA5_reanalyse_theta_all_levels_{day}_50N-90N.nc"

path_cldr = f"{root_input}/Data/{dataset}/ERA5_reanalyse_CDR_all_levels_{day}_50N-90N.nc"


var_u, var_v, var_w, var_t, var_p, var_theta, var_sp = 'u', 'v', 'w', 't', 'pres', 'theta', 'sp' #sometimes the variable for temperature is "temp" or "t"






u = xr.open_dataset(path_u, chunks=chunks)[var_u].rename({'valid_time': 'time'}).rename({'model_level': 'level'})  # .sel(longitude = slice(lon_min, lon_max)).sel(latitude = slice(lat_max, lat_min))
v = xr.open_dataset(path_v, chunks=chunks)[var_v].rename({'valid_time': 'time'}).rename({'model_level': 'level'})  # .sel(longitude = slice(lon_min, lon_max)).sel(latitude = slice(lat_max, lat_min))
w = xr.open_dataset(path_w, chunks=chunks)[var_w].rename({'valid_time': 'time'}).rename({'model_level': 'level'})  # .sel(longitude = slice(lon_min, lon_max)).sel(latitude = slice(lat_max, lat_min))
t = xr.open_dataset(path_t, chunks=chunks)[var_t].rename({'valid_time': 'time'}).rename({'model_level': 'level'})  # .sel(longitude = slice(lon_min, lon_max)).sel(latitude = slice(lat_max, lat_min))
p = xr.open_dataset(path_p, chunks=chunks)[var_p].rename({'valid_time': 'time'}).rename({'model_level': 'level'})  # Pa
theta = xr.open_dataset(path_theta, chunks=chunks)[var_theta].rename({'valid_time': 'time'}).rename({'model_level': 'level'})  # K


g = 9.8  # m.s-2
Rt = 6371e3  # m

print('uvwtptheta loaded', file = sys.stderr)

def f_coriolis(y):
    OMEGA = 7.29e-5  # rad.s-1
    return 2 * OMEGA * np.sin(y / 360 * 2 * np.pi)


dudy = u.differentiate(coord='latitude') / Rt * 360 / (2 * np.pi)
dudlevel = u.differentiate(coord='level')

dvdx = v.differentiate(coord='longitude') / (Rt * 2 * np.pi * np.cos(v['latitude'] / 360 * 2 * np.pi)) * 360
dvdlevel = v.differentiate(coord='level')

dthetadlevel = theta.differentiate(coord='level')
dthetady = theta.differentiate(coord='latitude') / Rt * 360 / (2 * np.pi)
dthetadx = theta.differentiate(coord='longitude') / (Rt * 2 * np.pi * np.cos(theta['latitude'] / 360 * 2 * np.pi)) * 360

vorticity_rel = dvdx - dudy + u / Rt * np.tan(u['latitude'] / 180 * np.pi)

fc = f_coriolis(u['latitude'])

vorticity_abs = vorticity_rel + fc

dpdlevel = p.differentiate(coord='level')  #if in hPa, multiply by 100 !!!

q_1 = -g / dpdlevel * vorticity_abs * dthetadlevel
q_2 = -g / dpdlevel * dudlevel * dthetady
q_3 = -g / dpdlevel * (- dvdlevel * dthetadx)

q = q_1 + q_2 + q_3
all_data = xr.Dataset(
   {
       'u': u ,
       'v': v ,
       'w': w ,
       'temp': t,
       'pv':   q,
       "pres": p,
   }
)




all_data.to_netcdf(f"{root_input}/Generated_data/data/{dataset}/PV/ERA5_reanalyse_u_v_w_temp_PV_pres_all_levels_2022-08-{day}_50N-90N.nc")

#(all_data.sel(latitude = slice(85,65), longitude = slice(0,80))).to_netcdf(f"{root_input}/Generated_data/data/{dataset}/PV/ERA5_forecasts_u_v_w_temp_PV_pres_all_levels_2022-08-{day}_65N-85N.nc")
#(all_data.sel(latitude = slice(lat_max,lat_min), longitude = slice(lon_min,lon_max))).to_netcdf(f"{root_input}/Generated_data/data/{dataset}/PV/ERA5_forecasts_u_v_w_temp_PV_pres_all_levels_2022-08-{day}_{lat_min}N-{lat_max}N.nc")


all_data.close()
# u.close()
# v.close()
# w.close()
# t.close()
# theta.close()
q.close()

print('all_data done', file = sys.stderr)

all_data.close()
q.close()
