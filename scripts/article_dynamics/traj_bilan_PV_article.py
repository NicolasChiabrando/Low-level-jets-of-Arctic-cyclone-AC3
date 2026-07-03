from tqdm import trange
import add_sys_path
add_sys_path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import csv
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import os
from mpl_toolkits.axes_grid1 import AxesGrid
from matplotlib.collections import LineCollection
from libs.interp import interp_traj



cyc = "AC3B" # "AC3A"

region = 0

if cyc != "AC3A" :
    region = 0

if region == 1:
    region_name = "PV zonal band"
elif region == 2:
    region_name = "Bent back front"
else :
    region_name = ""

txt_title = cyc
txt_save = ""

local = True #True

if cyc == 'AC3A' :
    niter=4
    dt_traj=0.5
    Nhor= 48#20
    Np=21 #11
    dataset = "era5_forecasts_1608T14_-15E-33E_78N-66.0N_750-960hPa" #"era5_forecasts_1608T14_5E-25E_78N-73.0N_850-960hPa"

    days = "15a16"
    loc = '65N-86N'
    time_ini=  '2022-08-16T14'
    long_min = -20
    long_max = 40
    lat_min = 66
    lat_max = 78
    level_min=1000
    level_max=800
    ini_time_step = 19
    h0 =14 #9 #intial hour
    traj_duration  =19
    txt_title = f"{region_name} - {cyc}"
    txt_save = f"_{region_name.replace(' ', '_')}"

elif cyc =='AC3B' :
    niter=4
    dt_traj=0.5
    Nhor=40
    Np=21
    dataset = "era5_forecasts_1908T14_-10E-30E_89N-79.0N_750-960hPa"
    days = "17a19"
    loc = '74N-90N'
    time_ini= '2022-08-19T14'
    long_min = -20  # -20
    long_max = 60
    lat_min = 75  # 65
    lat_max = 90

    level_min = 1000
    level_max = 800
    ini_time_step = 43 #31 #15 #7 #Initial time step of the trajectory 15 #
    h0 = 14 #intial hour
    lon_min_1 = -180  # -30
    lon_max_1 = 180  # 120
    lat_min_1 = 70
    lat_max_1 = 89.9
    traj_duration  =24

elif cyc == 'AC4' :
    niter=4
    dt_traj=0.5
    Nhor=40 #15
    Np=21
    dataset = "era5_forecasts_2308T09_10E-50E_80N-70.0N_750-960hPa" #"era5_forecasts_2308T09_25E-40E_78.5N-74.75N_850-960hPa"
    days = "22"
    loc = '65N-86N'
    time_ini=  '2022-08-23T09'
    long_min = 0
    long_max = 60
    lat_min = 67
    lat_max = 82
    level_min=1000
    level_max=800
    ini_time_step = 15 #31 #15 #7 #Initial time step of the trajectory 15 #
    h0 =9 #intial hour

if cyc == 'AC3A' :
    lon_min_3d = -30
    lon_max_3d = 40
    lat_max_3d = 85
    lat_min_3d = 65
elif cyc =='AC3B' :
    lon_min_3d = -30
    lon_max_3d = 70
    lat_max_3d = 88
    lat_min_3d = 78
elif cyc =='AC4' :
    lon_min_3d = 10
    lon_max_3d = 60
    lat_max_3d = 80
    lat_min_3d = 70

root_input = "/data/nchiab/PV"

if local :
    root_input = "/media/chabranoo/LaCie/PhD"






#%%
modlev = True #weither we are working with data in level model. Now it is by default True

hPa_pres = False  #weither the real pressure in the case of model levels (usually 'pres') are in hPa or not

P0 = 100000

if hPa_pres :
    P0 = 1000

name_p = 'P'

path_traj = f'{root_input}/Generated_data/data/era5_forecasts/Trajectories/{dataset}/Traj_time_step_{ini_time_step}_{traj_duration}h_dt{dt_traj}_niter{niter}_Nhor{Nhor}_Np{Np}.nc'
#path_traj = f'{root_input}/Generated_data/data/Trajectories/{dataset}/Traj_time_step_31_PM_model_levels.nc'
traj_ini = xr.open_dataset(path_traj)

dt = traj_ini['time'].isel(n_seeds=0, time_ind=0).data - traj_ini['time'].isel(n_seeds=0, time_ind=1).data
duration = traj_ini['time'].isel(n_seeds=0, time_ind=0).data - traj_ini['time'].isel(n_seeds=0, time_ind=-1).data
duration = 12
t_list = np.linspace(h0, h0 - duration, int(duration / dt) + 1, endpoint=True)

path_plots = f"{root_input}/Generated_data/plots/era5_forecasts/Trajectories/{dataset}/{duration}h/"

os.makedirs(path_plots, exist_ok=True)


path_data = f'{root_input}/Generated_data/data/era5_forecasts/ERA5_forecasts_all_pv_variables_all_levels_2022-08-{days}_{loc}.nc'
path_data_pressure =  f'{root_input}/Generated_data/data/era5_forecasts/ERA5_forecasts_all_pv_variables_700-1000hPa_per10hPa_2022-08-{days}_{loc}.nc'

#%%

if cyc =='AC3B' : #in order to remove the point at 90°N (divergence in PV)
    path_data = f'{root_input}/Generated_data/data/era5_forecasts/ERA5_forecasts_all_pv_variables_all_levels_2022-08-{days}_{loc}_all-long.nc'
    path_data_pressure = f'{root_input}/Generated_data/data/era5_forecasts/ERA5_forecasts_all_pv_variables_700-1000hPa_per10hPa_2022-08-{days}_{loc}_all-long.nc'
    all_data = xr.open_dataset(path_data).sel(longitude = slice(lon_min_1, lon_max_1)).sel(latitude = slice(lat_max_1, lat_min_1))
    all_data_pressure = xr.open_dataset(path_data_pressure).sel(longitude = slice(lon_min_1, lon_max_1)).sel(latitude = slice(lat_max_1, lat_min_1)).rename({"plev":"level"})
    Sp  = xr.open_dataset(f'{root_input}/Data/era5_forecasts/ERA5_SP_2022-08-{days}_{loc}.nc').sel(longitude = slice(lon_min_1, lon_max_1)).sel(latitude = slice(lat_max_1, lat_min_1))
else :
    all_data = xr.open_dataset(path_data)
    all_data_pressure = xr.open_dataset(path_data_pressure).rename({"plev":"level"})
    Sp  = xr.open_dataset(f'{root_input}/Data/era5_forecasts/ERA5_SP_2022-08-{days}_{loc}.nc')



all_data = all_data.sel(latitude=slice(None, None, -1))#.sel(level = slice(100000, 70000))

all_data_pressure = all_data_pressure.where(all_data_pressure['level'] < Sp['sp'], np.nan)

PV = all_data['PV_1'] + all_data['PV_2'] + all_data['PV_3']#.isel(time = slice(19, 32)).sel(latitude = slice(lat_max, lat_min), longitude = slice(long_min, long_max), level = slice(level_max, level_min))
PV_pressure = all_data_pressure['PV_1'] +  all_data_pressure['PV_2'] +   all_data_pressure['PV_3']
#%%
time = np.arange(0, np.size(PV['time']))
#%%

#%%

#traj_era5 = xr.open_dataset(path_traj_era5)
#data_lmdz = xr.open_dataset(path_lmdz)
n_date = PV['time'].isel(time = int(traj_ini['time'].isel(n_seeds = 0,time_ind = 0)))
date = f'{pd.to_datetime(n_date.data).year}-0{pd.to_datetime(n_date.data).month}-{pd.to_datetime(n_date.data).day}'
#%%


#n_list = traj_ini['n_seeds'].data
#n_list = []


#n_region1_list = []
#n_region2_list = []

from libs.traj import is_in_region

n_list = is_in_region(traj_ini, cyc)

#if region == 1:
#    n_list = n_region1_list
#elif region == 2:
#    n_list = n_region2_list

traj = traj_ini.sel(n_seeds = n_list).sel(time_ind = slice(0, int(duration  / dt + 1)))



print(f'Number of trajectories : {len(n_list)}')
#%%
#n_point = []

isnan_P = ~np.isnan(traj['P'])

n_point = np.sum(isnan_P, axis = 0)





dpvdt_traj = -(traj['pv'].differentiate('time_ind') /  (dt * 3600) * 1e6) # pvu / s
#rayon de la terre
mean_traj = traj.mean('n_seeds')

def theta_f(T, P, P0 = 1000) :
    return T * (P0 / P)**0.286
theta_traj = theta_f(traj['temp'], traj['P'], P0 = 100000)

pres = all_data["pres"]#.sel(level=slice(None, None, -1))

PV_dot_heating =all_data['PV_dot_heating_1'] + all_data['PV_dot_heating_2'] + all_data['PV_dot_heating_3']
PV_dot_friction =all_data['PV_dot_friction_1'] + all_data['PV_dot_friction_2'] + all_data['PV_dot_friction_3']
#%%
PV_dot = PV_dot_heating + PV_dot_friction

PV_dot_list = np.full((np.size(traj['n_seeds']),int(duration / dt) + 1), np.nan)
PV_dot_heating_list = np.full((np.size(traj['n_seeds']),int(duration / dt) + 1), np.nan)
PV_dot_friction_list = np.full((np.size(traj['n_seeds']),int(duration / dt) + 1), np.nan)


PV_dot_heating_SWR_list = np.full((np.size(traj['n_seeds']),int(duration / dt) + 1), np.nan)
PV_dot_heating_LWR_list = np.full((np.size(traj['n_seeds']),int(duration / dt) + 1), np.nan)
# PV_dot_heating_SWRCS_list = np.full((np.size(traj['n_seeds']),int(duration / dt) + 1), np.nan)
# PV_dot_heating_LWRCS_list = np.full((np.size(traj['n_seeds']),int(duration / dt) + 1), np.nan)

# n_seeds_pos = []




time = np.arange(0, PV_dot.shape[0])
level = PV_dot['level'].to_numpy()
lat = PV_dot['latitude'].to_numpy()
lon = PV_dot['longitude'].to_numpy()
# print(traj)
# print(np.shape(PV_dot_list))
if True :
    for i in trange(np.size(traj['n_seeds'].to_numpy()), desc="Processing"):
        # print(i)
        PV_dot_list[i] = interp_traj(i, PV_dot.to_numpy(), lat, lon, level, time, traj, varlev='m')

        PV_dot_heating_interp =  interp_traj(i, PV_dot_heating.to_numpy(), lat, lon, level, time, traj, varlev='m' )
        PV_dot_heating_list[i] =PV_dot_heating_interp

        PV_dot_friction_interp =  interp_traj(i, PV_dot_friction.to_numpy(), lat, lon, level, time, traj, varlev='m')
        PV_dot_friction_list[i] =PV_dot_friction_interp


        PV_dot_heating_SWR_interp = interp_traj(i, all_data["PV_dot_heating_SWR"].to_numpy(), lat, lon, level, time, traj, varlev='m')
        PV_dot_heating_SWR_list[i] = PV_dot_heating_SWR_interp
        PV_dot_heating_LWR_interp = interp_traj(i, all_data["PV_dot_heating_LWR"].to_numpy(), lat, lon, level, time, traj, varlev='m')
        PV_dot_heating_LWR_list[i] = PV_dot_heating_LWR_interp
        # PV_dot_heating_SWRCS_list[i] = interp_traj(i, all_data["PV_dot_heating_SWRCS"].to_numpy(), lat, lon, level, time, traj, varlev='m')
        # PV_dot_heating_LWRCS_list[i] = interp_traj(i, all_data["PV_dot_heating_LWRCS"].to_numpy(), lat, lon, level, time, traj, varlev='m')




    #     n_seeds_pos.append(i)
#%%
dpvdt_traj = -(traj['pv']).differentiate('time_ind') / (3600 * dt) * 1e6  # pvu / s
PV_dot_heating_rad_list = PV_dot_heating_SWR_list + PV_dot_heating_LWR_list

PV_dot_mean =  np.nanmean(PV_dot_list, axis = 0)
PV_dot_heating_mean =  np.nanmean(PV_dot_heating_list, axis = 0)
PV_dot_friction_mean =  np.nanmean(PV_dot_friction_list, axis = 0)



PV_dot_heating_SWR_mean = np.nanmean(PV_dot_heating_SWR_list, axis = 0)
PV_dot_heating_LWR_mean = np.nanmean(PV_dot_heating_LWR_list, axis = 0)
# PV_dot_heating_SWRCS_mean = np.nanmean(PV_dot_heating_SWRCS_list, axis = 0)
# PV_dot_heating_LWRCS_mean = np.nanmean(PV_dot_heating_LWRCS_list, axis = 0)
#%%
PV_heating_accumulated = np.zeros(np.size(mean_traj['time_ind']))
PV_heating_accumulated[-1] =   mean_traj['pv'][-1].values



PV_friction_accumulated = np.zeros(np.size(mean_traj['time_ind']))
PV_friction_accumulated[-1] =   mean_traj['pv'][-1].values




PV_friction_accumulated = np.zeros(np.size(mean_traj['time_ind']))
PV_friction_accumulated[-1] =   mean_traj['pv'][-1].values

PV_heating_SWR_accumulated = np.zeros(np.size(mean_traj['time_ind']))
PV_heating_SWR_accumulated[-1] = mean_traj['pv'][-1].values

PV_heating_LWR_accumulated = np.zeros(np.size(mean_traj['time_ind']))
PV_heating_LWR_accumulated[-1] = mean_traj['pv'][-1].values

# PV_heating_SWRCS_accumulated = np.zeros(np.size(mean_traj['time_ind']))
# PV_heating_SWRCS_accumulated[-1] = mean_traj['pv'][-1].values
#
# PV_heating_LWRCS_accumulated = np.zeros(np.size(mean_traj['time_ind']))
# PV_heating_LWRCS_accumulated[-1] = mean_traj['pv'][-1].values


var_u, var_v, var_w, var_t = 'u', 'v', 'w', 'temp'
u = all_data_pressure[var_u]  #
v = all_data_pressure[var_v]  #
w = all_data_pressure[var_w]  #
t = all_data_pressure[var_t]  #
p = all_data_pressure['pres']

pv_1 = all_data_pressure['PV_1']
pv_2 = all_data_pressure['PV_2']
pv_3 = all_data_pressure['PV_3']

pv = pv_1 + pv_2 + pv_3


for t in trange(1, np.size(mean_traj['time_ind'])) :
    PV_heating_accumulated[-1 -t] = PV_heating_accumulated[-1] + np.trapz(PV_dot_heating_mean[-1:-1-t:-1 ], x = traj.isel(n_seeds = 0)['time'][-1:-1-t:-1].values * 3600)

    PV_friction_accumulated[-1 -t] = PV_friction_accumulated[-1] + np.trapz(PV_dot_friction_mean[-1:-1-t:-1 ], x = traj.isel(n_seeds = 0)['time'][-1:-1-t:-1].values * 3600)

    PV_heating_SWR_accumulated[-1 -t] = PV_heating_SWR_accumulated[-1] + np.trapz(PV_dot_heating_SWR_mean[-1:-1-t:-1 ], x = traj.isel(n_seeds = 0)['time'][-1:-1-t:-1].values * 3600)

    PV_heating_LWR_accumulated[-1-t] = PV_heating_LWR_accumulated[-1] + np.trapz(PV_dot_heating_LWR_mean[-1:-1-t:-1 ], x = traj.isel(n_seeds = 0)['time'][-1:-1-t:-1].values * 3600)

    # PV_heating_SWRCS_accumulated[-1-t] = PV_heating_SWRCS_accumulated[-1] + np.trapz(PV_dot_heating_SWRCS_mean[-1:-1-t:-1 ], x = traj.isel(n_seeds = 0)['time'][-1:-1-t:-1].values * 3600)
    #
    # PV_heating_LWRCS_accumulated[-1-t] = PV_heating_LWRCS_accumulated[-1] + np.trapz(PV_dot_heating_LWRCS_mean[-1:-1-t:-1 ], x = traj.isel(n_seeds = 0)['time'][-1:-1-t:-1].values * 3600)



# Load dataset




from matplotlib.colors import LinearSegmentedColormap, ListedColormap
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


LON_traj = traj['lon']
LAT_traj = traj['lat']
P_traj = traj['P']
color = P_traj / 100

fig = plt.figure(figsize = (15,5))
# fig = plt.figure(figsize = (18 * 0.34, 8 * 0.34))

ax1 = plt.subplot(1,2,1, projection=ccrs.NorthPolarStereo())

ax1.scatter(LON_traj[::,0],LAT_traj[::,0], c=P_traj[::,0], edgecolors='black',
            cmap='Greens',transform=ccrs.PlateCarree())


im = (( all_data_pressure['PV_1'] +  all_data_pressure['PV_2'] +  all_data_pressure['PV_3']) * 1e6).isel(time = ini_time_step).sel(level = 950 * 100 , method = 'nearest' ).plot.contourf(ax=ax1, add_colorbar = False,  transform=ccrs.PlateCarree(), levels = pv_lev, cmap = newcmp) # levels =[-0.8, 0, 0.8, 1.6, 3.2, 4.8, 5.6])# )#, cmap = 'plasma')
#extent = 2500000
#ax1.set_extent((-extent,extent,-extent,extent),crs=ccrs.NorthPolarStereo())
#plt.title(f'Trajectories - {cyc} ', size=26)
ax1.set_extent([long_min, long_max, lat_max, lat_min], ccrs.PlateCarree())
ax1.coastlines(linewidth=0.2)
#plt.show()
for i_traj in traj['n_seeds'][::] :
    #print('itraj ==' + str(i_traj))#, file = sys.stderr)
    points = np.array([traj['lon'][i_traj,:], traj['lat'][i_traj,:]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    #norm = plt.Normalize(850,950)#
    norm = plt.Normalize(np.nanmin(color),np.nanmax(color))
    lc = LineCollection(segments, cmap='viridis', norm=norm,transform=ccrs.Geodetic(), alpha =0.3, zorder = 10)
    lc.set_array(color[i_traj,:])
    lc.set_linewidth(2)
    line = ax1.add_collection(lc)
ax1.set_xlim([(np.nanmin(LON_traj))-0.5,(np.nanmax(LON_traj))+0.5])
#print(np.nanmin(LON_traj)
ax1.set_ylim([(np.nanmin(LAT_traj))-0.5,(np.nanmax(LAT_traj))+0.5])

fig.subplots_adjust(bottom=0.2, wspace = 0.05)
cbar_ax = fig.add_axes([0.15, 0.1, 0.3, 0.04])
#cbar_ax = fig.add_axes([0.92, 0.125, 0.02, 0.755])

colo = fig.colorbar(lc, cax = cbar_ax,orientation='horizontal',pad = 0.11, shrink = 0.6)#, fraction=0.046) location = 'bottom',
# colo.ax.tick_params(labelsize=23)
colo.set_label(label='Pressure [hPa]')#, size=23)
colo.ax.invert_yaxis()
ax1.set_extent([long_min,long_max,lat_min,lat_max], ccrs.PlateCarree())
ax1.coastlines()
ax1.set_title(f'')
gl = ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
gl.bottom_labels = False
gl.right_labels = False

im = all_data_pressure['theta'].isel(time = ini_time_step).sel(level = 950 * 100 , method = 'nearest' ).plot.contour(ax=ax1,
        add_colorbar = False,  add_labels = False, transform=ccrs.PlateCarree(), colors ='black',  levels =  np.arange(260, 320, 2), zorder = 50)
plt.clabel(im, inline=True, colors = 'black', fontsize = 8)#, fontsize=8, fmt="%.2f")


ax2 = plt.subplot(1,2,2)
linewidth = 2.5
#plt.plot(t_list, PV_dot_mean * 1e6 * 3600, label = 'Interpolation total', c = 'black')
#plt.plot(t_list, dpvdt_traj.mean('n_seeds') , label = 'Derivative traj', , c = 'grey')
# plt.plot(t_list, PV_heating_accumulated * 1e6, label = 'Total heating', c = 'red')
ax2.plot(t_list, (PV_heating_SWR_accumulated + PV_heating_LWR_accumulated - mean_traj['pv'][-1].values) * 1e6 , label = 'rad', c = 'm' , linestyle = 'dotted', linewidth = linewidth)
# plt.plot(t_list, PV_heating_LWR_accumulated * 1e6 , label = 'LWR', c = "orange" )
ax2.plot(t_list,np.array(PV_friction_accumulated)  * 1e6 + np.array(PV_heating_accumulated) * 1e6 - np.array( mean_traj['pv'][-1].values) * 1e6, label = 'Heating + Friction' , c='black', linewidth = linewidth)
ax2.plot(t_list, mean_traj['pv'] * 1e6, label ='Full ¨PV', linestyle = 'dashed', c = 'grey', linewidth = linewidth)
ax2.plot(t_list, (PV_heating_accumulated  + PV_heating_LWR_accumulated[-1] - PV_heating_LWR_accumulated + PV_heating_SWR_accumulated[-1] - PV_heating_SWR_accumulated) * 1e6 , label = 'Heating - rad', c = 'green' , linestyle = 'dotted', linewidth = linewidth)
ax2.plot(t_list, PV_heating_accumulated * 1e6 , label=r'Heating', c='red', linewidth = linewidth)
ax2.plot(t_list, PV_friction_accumulated* 1e6 , label=r'Friction', c='blue', linewidth = linewidth)
# plt.plot(t_list, PV_heating_SWRCS_accumulated * 1e6 , label = 'SWRCS', linestyle = 'dotted'  )
# plt.plot(t_list, PV_heating_LWRCS_accumulated * 1e6 , label = 'LWRCS', linestyle = 'dotted'  )
#plt.plot(t_list, ( PV_dot_heatingSWR_mean + PV_dot_heatingLWR_mean + PV_dot_heatingSWRCS_mean +  PV_dot_heatingSWRCS_mean ) * 1e6 * 3600, linestyle = 'dotted', label='sum' )
ax2.legend()

ax2.set_ylabel('PV [pvu]')
ax2.set_xlabel('hours')
ax2.set_title(f'')
ax2.grid()
ax2.set_xlabel(f'hours on the {date}')


def trajtohours(x):
    return -(- x + 14)


def hourstotraj(x):
    return  (x - 14)

secax_x = ax2.secondary_xaxis('top', functions=(trajtohours, hourstotraj))
secax_x.set_xlabel(r'hours to initialisations')

def PV_lin(x):
    return x +  PV_heating_accumulated[-1]* 1e6


def delta_PV_fun(x):
    return x -  PV_heating_accumulated[-1] * 1e6

secax_y = ax2.secondary_yaxis('right', functions=(delta_PV_fun, PV_lin), color = 'orange')
secax_y.set_ylabel(r'$\Delta PV$')
ax2.hlines( mean_traj['pv'][-1] * 1e6, xmin = t_list[-1] -1 , xmax = t_list[0] +1 , color = 'orange', linestyle = 'dashed')
#plt.ylim([-0.0001, 0.0003])
#plt.ylim([-0.2, 0.6])
ax2.set_xlim([t_list[-1] -.2  , t_list[0] +.2])

label_list = ['a)', 'b)']
axs = [ax1, ax2]
for i in range(len(label_list)):
    axi = axs[i]
    axi.annotate(
        label_list[i],
        xy=(0, 1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize='large', verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))


plt.savefig(f'{path_plots}/traj_bilan_PV_{dataset}_{cyc}.png', bbox_inches='tight', transparent = True)
plt.show()






