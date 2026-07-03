# %%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import csv
import pandas as pd
import pint_xarray

import os

cyc = 'AC3A'

dataset = 'era5_forecasts'  # 'era5'

local = True #True  # 0 on spirit, 1 on lmd computer, 2 on personanl

root_input = "/data/nchiab/PV"

if local:
    root_input = "/media/chabranoo/LaCie/PhD"

path_plots = f"{root_input}/Generated_data/plots/{dataset}/{cyc}"

# %%

flight_name = 'F46'  #
path_flight = "/bdd/RALI-ThinIce/Data/RASTA/L2-wind-product_06.3.1/THINICE_20220816_F46_RASTA_WIND_final_06.3.1.nc"
day = "15a16"  #
ext = '65N-86N'  # '50N-90N' #
# lon_min =-15 # -5
# lon_max = 35
# lat_min = 65# 73
# lat_max = 78

lon_min = -20
lon_max = 40
lat_min = 66
lat_max = 78
# traj_name = "amip-ERA5-LAM-SVA30-extended1_20220801_20220831_HF_histhf_1608T14_dt0.5_niter4_5E-25E_850-950hPa_per10hPa"
# ini_time_step = 62
time_max = '2022-08-16T14'



if local:
    path_flight = f'/media/chabranoo/LaCie/PhD/data/rali/THINICE_{flight_name}_RASTA_WIND_final_06.3.1.nc'
path = f"{root_input}/Generated_data/data/{dataset}/ERA5_forecasts_all_pv_variables_700-1000hPa_per10hPa_2022-08-{day}_{ext}.nc"
# %%

all_data = xr.open_dataset(path).rename({"plev": "level"})
Sp = xr.open_dataset(f'{root_input}/Data/era5_forecasts/ERA5_SP_2022-08-{day}_{ext}.nc')

flight = xr.open_dataset(path_flight)

all_data = all_data.where(all_data['level'] < Sp['sp'], np.nan)


all_data = all_data.assign_coords(level=all_data.level / 100)

# update units metadata
all_data.level.attrs["units"] = "hPa"

# %%


path_data_ci=  f'{root_input}/Data/ERA5/ci.202208.as1e5.GLOBAL_025.nc'

ci = xr.open_dataset(path_data_ci)['siconc']#.sel(latitude = slice(lat_max, 50))

# %%
pres_hPa = False

if pres_hPa:
    fac_pres = 1
else:
    fac_pres = 100


# %%
def theta_func(T, P, P0=1000):
    return T * (P0 / P) ** 0.286


g = 9.8  # m.s-2
Rt = 6371e3  # m
# %%
kappa = 0.286  # R / Cp
theta_dot = all_data['theta_dot']
# %%
var_u, var_v, var_w, var_t = 'u', 'v', 'w', 'temp'
u = all_data[var_u]  #
v = all_data[var_v]  #
w = all_data[var_w]  #
t = all_data[var_t]  #
p = all_data['pres']

pv_1 = all_data['PV_1']
pv_2 = all_data['PV_2']
pv_3 = all_data['PV_3']

pv = pv_1 + pv_2 + pv_3
theta = all_data['theta']

# hello wo

# path_sp = f"{root_input}/Data/{dataset}/ERA5_SP_2022-08-{day}_50N-90N.nc"
# sp = xr.open_dataset(path_sp)['sp']
# PV = xr.open_dataset(path)['pv']
# %%
time = time_max

# %%
# g = 9.8
# prw = -1/g *all_data['ovap'].integrate(coord ="level")

# %% md
# Cartes


import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap




pres = 950 #950



uo_sampled = u.sel( time = time, level = pres, method = 'nearest'  )
vo_sampled = v.sel( time = time, level = pres, method = 'nearest')

# Weighting calculations
# Note: Ensure that 'latitude' values are in radians if converting from degrees
weights = np.cos(np.deg2rad(uo_sampled.latitude))
weights_matrix = np.tile(weights, (len(uo_sampled.longitude), 1)).T  # Adjust shape to match data

# Apply weights to the data (conceptual example - adjust as needed for your analysis)
uo_weighted = uo_sampled * weights_matrix
vo_weighted = vo_sampled * weights_matrix

# U2m = np.sqrt(uo_weighted**2 + vo_weighted**2).mean('time')

# uo_weighted = uo_weighted.mean('time')
# vo_weighted =vo_weighted.mean('time')
U = np.sqrt(uo_sampled**2 + vo_sampled**2)
UO = xr.merge([uo_weighted, vo_weighted])
# Create a figure and axis with your specified projection


fig = plt.figure(figsize = (30, 10))


proj = ccrs.NorthPolarStereo()  # Change projection as needed
ax = plt.subplot(1,1,1,projection=proj)

# Plot coastlines and set the global extent
ax.coastlines()
ax.set_global()
ax.set_extent([lon_min, lon_max, lat_min, lat_max], ccrs.PlateCarree())
gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

gl.xlabel_style = {'size': 20,}
gl.ylabel_style = {'size': 20,}
# Plot the unmasked topo data
# Plot the unmasked topo data


im = ci.mean('time').plot.contour(ax=ax, add_colorbar=False, levels = [0.8], cmap = 'c', transform=ccrs.PlateCarree() )
im = ci.mean('time').plot.contour(ax=ax, add_colorbar=False, levels = [0.2], cmap = 'blue', transform=ccrs.PlateCarree() )
im = theta.sel( time = time, level = pres).plot.contour(ax=ax, add_colorbar = False,  add_labels = False, transform=ccrs.PlateCarree(), colors ='black',  levels =  np.arange(260, 320, 2))
plt.clabel(im, inline=True, colors = 'black', fontsize = 12)#, fontsize=8, fmt="%.2f")

# Add other data variables
im = U.sel( method = 'nearest' ).plot.contourf(ax=ax, add_colorbar = False,  transform=ccrs.PlateCarree(), levels = np.arange(0,31,3))#,   extend = 'both') # levels =[-0.8, 0, 0.8, 1.6, 3.2, 4.8, 5.6])# )#, cmap = 'plasma')


cbar = fig.colorbar(im, ax=ax, orientation ='horizontal', fraction=0.046, pad=0.1)
cbar.set_label(rf'{pres} hPa horizontal wind speed [m.s$^{-1}$]', fontsize = 14)
cbar.ax.tick_params(labelsize=13)
#im = (dthetadx ).sel( time = time, level = pres * fac_pres ).plot.contourf(ax=ax, add_colorbar = False,  transform=ccrs.PlateCarree())#, levels = np.arange(-10, 10, 1)) # levels =[-0.8, 0, 0.8, 1.6, 3.2, 4.8, 5.6])# )#, cmap = 'plasma')

# from matplotlib.collections import LineCollection
#
# points = np.array([flight['longitude'], flight['latitude']]).T.reshape(-1, 1, 2)
# segments = np.concatenate([points[:-1], points[1:]], axis=1)
# # norm = plt.Normalize(850,950)#
# norm = plt.Normalize(np.nanmin(flight['altitude']), np.nanmax((flight['altitude'])))
# lc = LineCollection(segments,  norm=norm, cmap = 'plasma', transform=ccrs.Geodetic())
# lc.set_array(flight['altitude'])
# lc.set_linewidth(2)
# line = ax.add_collection(lc)
# colo = fig.colorbar(lc, ax = ax,location = 'right', pad = 0.11, shrink = 0.8)#, fraction=0.046)
# # colo.ax.tick_params(labelsize=23)
# colo.set_label(label='Pressure [hPa]')#, size=23)

ax.plot(flight['longitude'], flight['latitude'], c = 'm', label = flight_name + ' trajectory',  transform=ccrs.PlateCarree(),  )

UO.plot.quiver(x = 'longitude', y = 'latitude', u = var_u, v = var_v, ax = ax, regrid_shape = 15,
                                    transform = ccrs.PlateCarree(), add_guide = False, color = 'grey')
ax.legend(fontsize= 20, loc = 'lower right')
ax.set_title('')





plt.savefig(f'{path_plots}/U&theta&seaice_{int(pres)}_{time}_{cyc}_{dataset}.png', bbox_inches='tight', transparent=True, dpi=300)

plt.show()












# plt.savefig(f"{path_plots}/PV_U_theta_{int(pres / 100)}hPa_{time}_{dataset}.png",  transparent=True)
