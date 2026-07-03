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
dataset = 'era5_forecasts'  # 'era5'
day = "15a16" #"15a16" # "15a16" #"22" #
lon_min = -30
lon_max = 70

lat_min = 65
lat_max = 86



local = False
chunks = {'valid_time': 3} # 'model_level': 1
root_input = "/data/nchiab/PV"

if local:
    root_input = "/media/chabranoo/LaCie/PhD"




path_u = f"{root_input}/Data/{dataset}/ERA5_U_all_levels_2022-08-{day}_50N-90N.nc"
path_v = f"{root_input}/Data/{dataset}/ERA5_V_all_levels_2022-08-{day}_50N-90N.nc"
path_w = f"{root_input}/Data/{dataset}/ERA5_Omega_all_levels_2022-08-{day}_50N-90N.nc"
path_t = f"{root_input}/Data/{dataset}/ERA5_TEMP_all_levels_2022-08-{day}_50N-90N.nc"
path_p = f"{root_input}/Data/{dataset}/ERA5_pres_all_levels_2022-08-{day}_50N-90N.nc"
# path_sp = f"{root_input}/Data/{dataset}/ERA5_SP_2022-08-{day}_50N-90N.nc"
path_theta = f"{root_input}/Data/{dataset}/ERA5_theta_all_levels_2022-08-{day}_50N-90N.nc"
path_ovap = f"{root_input}/Data/{dataset}/ERA5_OVAP_all_levels_2022-08-{day}_50N-90N.nc" #sueuERA5_OVAP_all_levels_2022-08-22_18:00:00_50N-90N.nc" #
path_geop = f"{root_input}/Data/{dataset}/ERA5_GEOP_all_levels_2022-08-{day}_50N-90N.nc"
path_cldr = f"{root_input}/Data/{dataset}/ERA5_CDR_all_levels_2022-08-{day}_50N-90N.nc"
path_lwc = f"{root_input}/Data/{dataset}/ERA5_lwc_all_levels_2022-08-{day}_50N-90N.nc"
path_iwc = f"{root_input}/Data/{dataset}/ERA5_iwc_all_levels_2022-08-{day}_50N-90N.nc"

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




#all_data.to_netcdf(f"{root_input}/Generated_data/data/{dataset}/PV/ERA5_forecasts_u_v_w_temp_PV_pres_all_levels_2022-08-{day}_50N-90N.nc")

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

kappa = 0.286 # R / Cp


path_MTTPM  = f"{root_input}/Data/{dataset}/ERA5_MTTPM_all_levels_2022-08-{day}_50N-90N.nc"
path_MUTPM = f"{root_input}/Data/{dataset}/ERA5_MUTPM_all_levels_2022-08-{day}_50N-90N.nc"
path_MVTPM = f"{root_input}/Data/{dataset}/ERA5_MVTPM_all_levels_2022-08-{day}_50N-90N.nc"

path_MTTLWR  = f"{root_input}/Data/{dataset}/ERA5_MTTLWR_all_levels_2022-08-{day}_50N-90N.nc"
path_MTTSWR  = f"{root_input}/Data/{dataset}/ERA5_MTTSWR_all_levels_2022-08-{day}_50N-90N.nc"
path_MTTLWRCS  = f"{root_input}/Data/{dataset}/ERA5_MTTLWRCS_all_levels_2022-08-{day}_50N-90N.nc"
path_MTTSWRCS  = f"{root_input}/Data/{dataset}/ERA5_MTTSWRCS_all_levels_2022-08-{day}_50N-90N.nc"

MTTPM = xr.open_dataset(path_MTTPM, chunks=chunks)['avg_ttpm'].rename({'valid_time': 'time'}).rename({'model_level': 'level'})#.sel(longitude = slice(lon_min, lon_max)).sel(latitude = slice(lat_max, lat_min))
MTTLWR = xr.open_dataset(path_MTTLWR, chunks=chunks)['avg_ttlwr'].rename({'valid_time': 'time'}).rename({'model_level': 'level'})#.sel(longitude = slice(lon_min, lon_max)).sel(latitude = slice(lat_max, lat_min))
MTTSWR = xr.open_dataset(path_MTTSWR, chunks=chunks)['avg_ttswr'].rename({'valid_time': 'time'}).rename({'model_level': 'level'})#.sel(longitude = slice(lon_min, lon_max)).sel(latitude = slice(lat_max, lat_min))
MTTLWRCS = xr.open_dataset(path_MTTLWRCS, chunks=chunks)['avg_ttlwrcs'].rename({'valid_time': 'time'}).rename({'model_level': 'level'})#.sel(longitude = slice(lon_min, lon_max)).sel(latitude = slice(lat_max, lat_min))
MTTSWRCS = xr.open_dataset(path_MTTSWRCS, chunks=chunks)['avg_ttswrcs'].rename({'valid_time': 'time'}).rename({'model_level': 'level'})#.sel(longitude = slice(lon_min, lon_max)).sel(latitude = slice(lat_max, lat_min))

Fx = xr.open_dataset(path_MUTPM, chunks=chunks)['avg_utpm'].rename({'valid_time': 'time'}).rename({'model_level': 'level'})#.sel(longitude = slice(lon_min, lon_max)).sel(latitude = slice(lat_max, lat_min))

Fy = xr.open_dataset(path_MVTPM, chunks=chunks)['avg_vtpm'].rename({'valid_time': 'time'}).rename({'model_level': 'level'})#.sel(longitude = slice(lon_min, lon_max)).sel(latitude = slice(lat_max, lat_min))


theta_dot = MTTPM * (100000 / p)**kappa #/ Cp
theta_dot_LWR = MTTLWR * (100000 / p)**kappa #/ Cp
theta_dot_SWR = MTTSWR * (100000 / p)**kappa #/ Cp
theta_dot_LWRCS = MTTLWRCS * (100000 / p)**kappa #/ Cp
theta_dot_SWRCS = MTTSWRCS * (100000 / p)**kappa #/ Cp

dtheta_dotdlevel = theta_dot.differentiate(coord = 'level')
dtheta_dotdy = theta_dot.differentiate(coord = 'latitude') / Rt * 360 / (2 * np.pi)
dtheta_dotdx = theta_dot.differentiate(coord = 'longitude')  / (Rt * 2 * np.pi * np.cos(theta_dot['latitude'] / 360 * 2 * np.pi )) * 360

dtheta_dot_LWRdlevel = theta_dot_LWR.differentiate(coord = 'level')
dtheta_dot_LWRdy = theta_dot_LWR.differentiate(coord = 'latitude') / Rt * 360 / (2 * np.pi)
dtheta_dot_LWRdx = theta_dot_LWR.differentiate(coord = 'longitude')  / (Rt * 2 * np.pi * np.cos(theta_dot_LWR['latitude'] / 360 * 2 * np.pi )) * 360

dtheta_dot_SWRdlevel = theta_dot_SWR.differentiate(coord = 'level')
dtheta_dot_SWRdy = theta_dot_SWR.differentiate(coord = 'latitude') / Rt * 360 / (2 * np.pi)
dtheta_dot_SWRdx = theta_dot_SWR.differentiate(coord = 'longitude')  / (Rt * 2 * np.pi * np.cos(theta_dot_SWR['latitude'] / 360 * 2 * np.pi )) * 360

dtheta_dot_LWRCSdlevel = theta_dot_LWRCS.differentiate(coord = 'level')
dtheta_dot_LWRCSdy = theta_dot_LWRCS.differentiate(coord = 'latitude') / Rt * 360 / (2 * np.pi)
dtheta_dot_LWRCSdx = theta_dot_LWRCS.differentiate(coord = 'longitude')  / (Rt * 2 * np.pi * np.cos(theta_dot_LWRCS['latitude'] / 360 * 2 * np.pi )) * 360

dtheta_dot_SWRCSdlevel = theta_dot_SWRCS.differentiate(coord = 'level')
dtheta_dot_SWRCSdy = theta_dot_SWRCS.differentiate(coord = 'latitude') / Rt * 360 / (2 * np.pi)
dtheta_dot_SWRCSdx = theta_dot_SWRCS.differentiate(coord = 'longitude')  / (Rt * 2 * np.pi * np.cos(theta_dot_SWRCS['latitude'] / 360 * 2 * np.pi )) * 360


dFxdlevel = Fx.differentiate(coord = 'level')
dFxdy = Fx.differentiate(coord = 'latitude') / Rt* 360 / (2 * np.pi)

dFydlevel = Fy.differentiate(coord = 'level')
dFydx = Fy.differentiate(coord = 'longitude')  / (Rt * 2 * np.pi * np.cos(Fy['latitude'] / 360* 2 * np.pi )) * 360

dqdt_heating = -g / dpdlevel * (vorticity_abs * dtheta_dotdlevel + dudlevel * dtheta_dotdy - dvdlevel * dtheta_dotdx )

dqdt_friction = - g / dpdlevel * ((dFydx - dFxdy + Fx * np.tan(Fx['latitude'] / 180 * np.pi) / Rt ) * dthetadlevel + dFxdlevel * dthetady - dFydlevel * dthetadx)

dqdt_heating_LWR = - g / dpdlevel * (vorticity_abs * dtheta_dot_LWRdlevel + dudlevel * dtheta_dot_LWRdy - dvdlevel * dtheta_dot_LWRdx)
dqdt_heating_SWR = - g / dpdlevel * (vorticity_abs * dtheta_dot_SWRdlevel + dudlevel * dtheta_dot_SWRdy - dvdlevel * dtheta_dot_SWRdx)
dqdt_heating_LWRCS = - g / dpdlevel * (vorticity_abs * dtheta_dot_LWRCSdlevel + dudlevel * dtheta_dot_LWRCSdy - dvdlevel * dtheta_dot_LWRCSdx)
dqdt_heating_SWRCS = - g / dpdlevel * (vorticity_abs * dtheta_dot_SWRCSdlevel + dudlevel * dtheta_dot_SWRCSdy - dvdlevel * dtheta_dot_SWRCSdx)

dqdt = dqdt_heating + dqdt_friction
dqdt.name = 'PV_dot'
dqdt_heating.name = 'PV_dot_heating'
dqdt_friction.name = 'PV_dot_friction'
dqdt_heating_LWR.name = 'PV_dot_heating_LWR'
dqdt_heating_SWR.name = 'PV_dot_heating_SWR'
dqdt_heating_LWRCS.name = 'PV_dot_heating_LWRCS'
dqdt_heating_SWRCS.name = 'PV_dot_heating_SWRCS'


ovap = xr.open_dataset(path_ovap, chunks=chunks)['q'].rename({'valid_time': 'time'}).rename({'model_level': 'level'})  # K
geop = xr.open_dataset(path_geop, chunks=chunks)['z'].rename({'valid_time': 'time'}).rename({'model_level': 'level'})  # m
cldr = xr.open_dataset(path_cldr, chunks=chunks)['cc'].rename({'valid_time': 'time'}).rename({'model_level': 'level'})  # K
lwc = xr.open_dataset(path_lwc, chunks=chunks)['clwc'].rename({'valid_time': 'time'}).rename({'model_level': 'level'})  # kg/kg
iwc = xr.open_dataset(path_iwc, chunks=chunks)['ciwc'].rename({'valid_time': 'time'}).rename({'model_level': 'level'})

all_variables = xr.Dataset(
    {
        "u": u,
        "v": v,
        "w": w,
        "temp": t,
        'theta': theta,
        "theta_dot" : theta_dot,
        "pres": p,
        "PV_dot_friction_1": -g / dpdlevel * (dFydx - dFxdy + Fx * np.tan(Fx['latitude'] / 180 * np.pi) / Rt) * dthetadlevel,
        "PV_dot_friction_2": -g / dpdlevel * dFxdlevel * dthetady,
        "PV_dot_friction_3": -g / dpdlevel * (- dFydlevel * dthetadx),
        "PV_dot_friction_Fx":  -g / dpdlevel * (- dFxdy + Fx * np.tan(Fx['latitude'] / 180 * np.pi) / Rt) * dthetadlevel + - g / dpdlevel * dFxdlevel * dthetady,
        "PV_dot_friction_Fy":  -g / dpdlevel * dFydx * dthetadlevel + g / dpdlevel * dFydlevel * dthetadx,
        "PV_dot_heating_1": -g / dpdlevel * vorticity_abs * dtheta_dotdlevel,
        "PV_dot_heating_2": -g / dpdlevel * dudlevel * dtheta_dotdy,
        "PV_dot_heating_3": -g / dpdlevel * (- dvdlevel * dtheta_dotdx),
        "PV_dot_heating_LWR": dqdt_heating_LWR,
        "PV_dot_heating_SWR": dqdt_heating_SWR,
        "PV_dot_heating_LWRCS": dqdt_heating_LWRCS,
        "PV_dot_heating_SWRCS": dqdt_heating_SWRCS,
        "PV_1" : q_1,
        "PV_2" : q_2,
        "PV_3" : q_3,
        "MTTPM": MTTPM,
        "MTTLWR": MTTLWR,
        "MTTSWR": MTTSWR,
        "MTTLWRCS": MTTLWRCS,
        "MTTSWRCS": MTTSWRCS,
        "Fx": Fx,
        "Fy": Fy,
        "ovap": ovap,
        "geop": geop,
        "cldr": cldr,
        "ciwc" : iwc,
        "clwc" : lwc
            }
)

all_variables.sel(latitude = slice(lat_max,lat_min), longitude = slice(lon_min,lon_max)).to_netcdf(f"{root_input}/Generated_data/data/{dataset}/ERA5_forecasts_all_pv_variables_all_levels_2022-08-{day}_{lat_min}N-{lat_max}N.nc")
