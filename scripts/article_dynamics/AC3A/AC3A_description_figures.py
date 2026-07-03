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
# lon_min = -15
# lon_max = 35
# lat_min = 65
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
pv_lev = [-0.2,0,0.2,.4,.6,.8,1,1.5,2,3,4,5,6]


import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
# hot = mpl.colormaps['OrRd'].resampled(np.size(pv_lev))
#
# newcolors = hot(np.linspace(0, 1, np.size(pv_lev)))
#
# blue = np.array([0, 0.4, 0.6, 1])
# newcolors[0, :] = blue
# newcmp = ListedColormap(newcolors)

pv_data = [ # 14 colors
     [76, 76, 248],   #blue
     [145, 172, 243],
     [211, 228, 235],
     [255, 210, 157],
     [223, 158, 90],
     [249, 0, 0],      #red
     [250, 122, 0],
     [249, 248, 7],
     [120, 221, 29],
     [121, 188, 85] #green
]

pv_data_norm = [[r/255, g/255, b/255] for r, g, b in pv_data]

# Create colormap


newcmp = ListedColormap(pv_data_norm, name = 'cmap_pv')

pv_lev =  [0,.5,1,1.5,2,3,5,8,10]


pres = 950 #950

lon_min_cs = 20  # 0 #0#28 #
lon_max_cs = 25  # 20 # 20  #50 #


# lon_min_cs = 5  # 0 #0#28 #
# lon_max_cs = 10  # 20 # 20  #50 #

lat_min_cs = 72
lat_max_cs = 78
lev_min_cs = 1050
lev_max_cs = 400

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
# fig = plt.figure(figsize = (20, 30))



proj = ccrs.NorthPolarStereo()  # Change projection as needed
ax = plt.subplot(1,2,1,projection=proj)
# ax = plt.subplot(2,1,1,projection=proj)
# Plot coastlines and set the global extent
ax.coastlines()
ax.set_global()
ax.set_extent([lon_min, lon_max, lat_min, lat_max], ccrs.PlateCarree())
gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

gl.xlabel_style = {'size': 20,}
gl.ylabel_style = {'size': 20,}
# Plot the unmasked topo data
# Plot the unmasked topo data



im = theta.sel( time = time, level = pres).plot.contour(ax=ax, add_colorbar = False,  add_labels = False, transform=ccrs.PlateCarree(), colors ='black', linewidth = 3, levels =  np.arange(260, 320, 2))
plt.clabel(im, inline=True, colors = 'black', fontsize = 15)#, fontsize=8, fmt="%.2f")

# Add other data variables
im = (pv * 1e6).sel( time = time, level = pres, method = 'nearest' ).plot.contourf(ax=ax, add_colorbar = False,  transform=ccrs.PlateCarree(), levels = pv_lev, cmap = newcmp,   extend = 'both') # levels =[-0.8, 0, 0.8, 1.6, 3.2, 4.8, 5.6])# )#, cmap = 'plasma')

#im = (dthetadx ).sel( time = time, level = pres * fac_pres ).plot.contourf(ax=ax, add_colorbar = False,  transform=ccrs.PlateCarree())#, levels = np.arange(-10, 10, 1)) # levels =[-0.8, 0, 0.8, 1.6, 3.2, 4.8, 5.6])# )#, cmap = 'plasma')

ax.plot(flight['longitude'], flight['latitude'], label = flight_name,  transform=ccrs.PlateCarree(), color = 'cyan', )

UO.plot.quiver(x = 'longitude', y = 'latitude', u = var_u, v = var_v, ax = ax, regrid_shape = 15,
                                    transform = ccrs.PlateCarree(), add_guide = False, color = 'grey')
# CS = U.plot.contour(ax=ax, add_colorbar = False,  transform=ccrs.PlateCarree(), cmap ='green', levels = np.arange(0,20, 5))
# plt.clabel(CS, inline=True, colors = 'black')#, fontsize=8, fmt="%.2f")

ax.plot(np.linspace(lon_min_cs, lon_min_cs, 10), np.linspace(lat_min_cs, lat_max_cs, 10), color = 'red', transform=ccrs.Geodetic(), linewidth = 3)
ax.plot(np.linspace(lon_min_cs, lon_max_cs, 10), np.linspace( lat_max_cs, lat_max_cs, 10), color = 'red', transform=ccrs.Geodetic(), linewidth = 3)
ax.plot(np.linspace( lon_max_cs, lon_max_cs, 10), np.linspace( lat_max_cs,lat_min_cs, 10), color = 'red', transform=ccrs.Geodetic(), linewidth = 3)
ax.plot(np.linspace(lon_max_cs, lon_min_cs, 10), np.linspace( lat_min_cs, lat_min_cs, 10), color = 'red', transform=ccrs.Geodetic(), linewidth = 3)

CS = U.plot.contour(ax=ax, add_colorbar = False,  transform=ccrs.PlateCarree(), cmap ='green', levels = np.arange(15,31, 5), linewidths = 3)
plt.clabel(CS, inline=True, colors = 'green', fontsize = 15)#, fmt = "%.2f")
ax.legend()
ax.set_title('')




ax = plt.subplot(1,2,2)
# ax = plt.subplot(2,1,2)
# Plot coastlines and set the global extent
#ax.coastlines()
#ax.set_global()
#ax.set_extent([lon_min, lon_max, lat_min, lat_max], ccrs.PlateCarree())
#ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

# Plot the unmasked topo data


# Add other data variables
im = (pv * 1e6).sel(time=time,  method='nearest').sel(longitude=slice(lon_min_cs,lon_max_cs),level=slice(lev_min_cs, lev_max_cs),
                                                                    latitude=slice(lat_max_cs, lat_min_cs)).mean('longitude', skipna = False).plot.contourf(
    ax = ax, add_colorbar = False,levels = pv_lev, cmap = newcmp, extend = 'both')  # levels =[-0.8, 0, 0.8, 1.6, 3.2, 4.8, 5.6])# )#, cmap = 'plasma')
# np.arange(-4, 4.1, 1.)
#im = (dthetadx ).sel( time = time, level = pres * fac_pres ).plot.contourf(ax=ax, add_colorbar = False,  transform=ccrs.PlateCarree())#, levels = np.arange(-10, 10, 1)) # levels =[-0.8, 0, 0.8, 1.6, 3.2, 4.8, 5.6])# )#, cmap = 'plasma')



CSu = u.sel(time=time,  method='nearest').sel(longitude=slice(lon_min_cs,lon_max_cs),level=slice(lev_min_cs, lev_max_cs),
                                                                    latitude=slice(lat_max_cs, lat_min_cs)).mean('longitude', skipna = False).plot.contour(ax=ax,
                                                                                                                  add_colorbar=False,
                                                                                                                  colors='green', levels=  np.arange(-24,25,4), linewidth = 3)



CSt = theta.sel(time=time,  method='nearest').sel(longitude=slice(lon_min_cs,lon_max_cs),level=slice(lev_min_cs, lev_max_cs),
                                                                    latitude=slice(lat_max_cs, lat_min_cs)).mean('longitude', skipna = False).plot.contour(ax=ax,
                                                                                                                  add_colorbar=False,
                                                                                                                  colors='black', linewidth = 3,  levels=  np.arange(260,320,2))
ax.hlines(pres, lat_min_cs, lat_max_cs, linestyle = 'dashed', color = 'red')
#ax.clabel(CSt)
#ax.clabel(CSu)

# set yticks from Pa to hPa




ax.invert_yaxis()

# cbar = fig.colorbar(im, ax=ax, orientation ='vertical', fraction=0.046, pad=0.04)
# cbar.set_label('PV [pvu]', fontsize =20)


plt.clabel(CSt, inline=True, colors='black', fontsize=15)#, fmt="%.2f")
plt.clabel(CSu, inline=True, colors='green', fontsize=15)#, fmt="%.2f")
plt.title('')
plt.ylabel('Pressure [hPa]', fontsize=20)
plt.xlabel('Latitude [°]', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)
# plt.savefig(f"{path_plots}/PV_u_theta_CS_mean{lon_min_cs}E-{lon_max_cs}E_{time}_{dataset}.png",  transparent=True)
# fig.tight_layout()
fig.subplots_adjust(bottom=0.2, wspace = 0.15)

cbar_ax = fig.add_axes([0.3, 0.07, 0.4, 0.05])
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')#,label = 'Potential vorticity [pvu]')#, fontsize = 20)

# fig.subplots_adjust(right=0.85, hspace = 0.1)
# cbar_ax = fig.add_axes([0.9, 0.3, 0.04, 0.4])
# cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')#,label = 'Potential vorticity [pvu]')#, fontsize = 20)
ticks =pv_lev # [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 3, 4, 5, 6]

cbar.set_ticks(ticks)
cbar.set_ticklabels(['0','.5','1','1.5','2','3','5','8','10'])
# cbar.set_ticklabels(['-0.2','0','0.2','0.4','0.6','0.8','1','1.5','2','3','4','5','6'])
cbar.set_label('PV [pvu]', size = 20)
cbar.ax.tick_params(labelsize=20)


label_list = ['a)', 'b)']

for i in range(len(label_list)):
    axi =  plt.subplot(1, 2, i +1)
    # axi =  plt.subplot(2,1,i +1)
    axi.annotate(
        label_list[i],
        xy=(0, 1), xycoords='axes fraction',
        xytext = (+0.5, -0.5), textcoords='offset fontsize',
        fontsize=25, verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))

plt.savefig(f'{path_plots}/PV_U_theta_hPa_hor&CS_{int(pres)}_{time}_{cyc}_{dataset}.png', bbox_inches='tight', transparent=True, dpi=300)

plt.show()












# plt.savefig(f"{path_plots}/PV_U_theta_{int(pres / 100)}hPa_{time}_{dataset}.png",  transparent=True)
