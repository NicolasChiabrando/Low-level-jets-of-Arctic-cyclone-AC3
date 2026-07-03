from tqdm import trange
import add_sys_path
add_sys_path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from mpl_toolkits.axes_grid1 import AxesGrid
from matplotlib.collections import LineCollection
from libs.interp import interp_traj
from mpl_toolkits.mplot3d import Axes3D
import os


test = 0

time_ind = 0
region = 0


cyc = "AC3A" # "AC3A"

txt_title = cyc
txt_save = ""

local = True

if cyc == 'AC3A' :
    niter=4
    dt_traj=0.5
    Nhor= 48#20
    Np=21 #11
    dataset = "era5_forecasts_1608T14_-15E-33E_78N-66.0N_750-960hPa" #"era5_forecasts_1608T14_5E-25E_78N-73.0N_850-960hPa"

    days = "15a16"
    loc = '65N-86N'
    time_ini=  '2022-08-16T14'
    long_min = -15
    long_max = 35
    lat_min = 65
    lat_max = 82
    level_min=1000
    level_max=800
    ini_time_step = 19
    h0 =14 #9 #intial hour
    traj_duration  =19
    # txt_title = f"{region_name} - {cyc}"
    # txt_save = f"_{region_name.replace(' ', '_')}"

elif cyc =='AC3B' :
    niter=4
    dt_traj=0.5
    Nhor=40
    Np=21
    dataset = "era5_forecasts_1908T14_-10E-30E_89N-79.0N_750-960hPa"
    days = "17a19"
    loc = '74N-90N'
    time_ini= '2022-08-19T14'
    long_min = -20 #-20
    long_max = 60
    lat_min = 74 #65
    lat_max = 90

    level_min = 1000
    level_max = 800
    ini_time_step = 43 #31 #15 #7 #Initial time step of the trajectory 15 #
    h0 = 14 #intial hour
    lon_min_1 = -30
    lon_max_1 = 120
    lat_min_1 = 75
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


root_input = "/data/nchiab/PV"

if local :
    root_input = "/media/chabranoo/LaCie/PhD"



path_plots = f"{root_input}/Generated_data/plots/era5_forecasts/"

if region != 0 :
    path_plots = f"{root_input}/Generated_data/plots/era5_forecasts/"



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


path_data = f'{root_input}/Generated_data/data/era5_forecasts/ERA5_forecasts_all_pv_variables_all_levels_2022-08-{days}_{loc}.nc'
path_data_pressure =  f'{root_input}/Generated_data/data/era5_forecasts/ERA5_forecasts_all_pv_variables_700-1000hPa_per10hPa_2022-08-{days}_{loc}.nc'


if cyc =='AC3B' : #in order to remove the point at 90°N (divergence in PV)
    all_data = xr.open_dataset(path_data).sel(longitude = slice(lon_min_1, lon_max_1)).sel(latitude = slice(lat_max_1, lat_min_1))
    all_data_pressure = xr.open_dataset(path_data_pressure).sel(longitude = slice(lon_min_1, lon_max_1)).sel(latitude = slice(lat_max_1, lat_min_1)).rename({"plev":"level"})
    Sp  = xr.open_dataset(f'{root_input}/Data/era5_forecasts/ERA5_SP_2022-08-{days}_{loc}.nc').sel(longitude = slice(lon_min_1, lon_max_1)).sel(latitude = slice(lat_max_1, lat_min_1))
else :
    all_data = xr.open_dataset(path_data)
    all_data_pressure = xr.open_dataset(path_data_pressure).rename({"plev":"level"})
    Sp  = xr.open_dataset(f'{root_input}/Data/era5_forecasts/ERA5_SP_2022-08-{days}_{loc}.nc')


mask = xr.open_dataset("/media/chabranoo/LaCie/PhD/Data/IMERG_land_sea_mask.nc")
mask = mask.rename({'lon': 'longitude', 'lat': 'latitude'}) #.expand_dims({'level': all_data_pressure['level'] / 100})
#
#

all_data = all_data.sel(latitude=slice(None, None, -1))#.sel(level = slice(100000, 70000))

all_data_pressure = all_data_pressure.where(all_data_pressure['level'] < Sp['sp'], np.nan)

PV = all_data['PV_1'] + all_data['PV_2'] + all_data['PV_3']#.isel(time = slice(19, 32)).sel(latitude = slice(lat_max, lat_min), longitude = slice(long_min, long_max), level = slice(level_max, level_min))
PV_pressure = all_data_pressure['PV_1'] +  all_data_pressure['PV_2'] +   all_data_pressure['PV_3']
#%%
time = np.arange(0, np.size(PV['time']))
#%%

n_date = PV['time'].isel(time = int(traj_ini['time'].isel(n_seeds = 0,time_ind = 0)))
date = f'{pd.to_datetime(n_date.data).year}-{pd.to_datetime(n_date.data).month}-{pd.to_datetime(n_date.data).day}'
#%%


from libs.traj import is_in_region

n_list = is_in_region(traj_ini, cyc)


traj = traj_ini.sel(n_seeds = n_list).sel(time_ind = slice(0, int(duration  / dt + 1)))


isnan_P = ~np.isnan(traj['P'])

n_point = np.sum(isnan_P, axis = 0)


mean_traj = traj.mean('n_seeds')


PV_dot_heating =all_data['PV_dot_heating_1'] + all_data['PV_dot_heating_2'] + all_data['PV_dot_heating_3']
PV_dot_friction =all_data['PV_dot_friction_1'] + all_data['PV_dot_friction_2'] + all_data['PV_dot_friction_3']
#%%
PV_dot = PV_dot_heating + PV_dot_friction

PV_dot_list = np.full((np.size(traj['n_seeds']),int(duration / dt) + 1), np.nan)
PV_dot_heating_list = np.full((np.size(traj['n_seeds']),int(duration / dt) + 1), np.nan)
PV_dot_friction_list = np.full((np.size(traj['n_seeds']),int(duration / dt) + 1), np.nan)

PV_dot_friction_1_list = np.full((np.size(traj['n_seeds']),int(duration / dt) + 1), np.nan)
PV_dot_friction_2_list = np.full((np.size(traj['n_seeds']),int(duration / dt) + 1), np.nan)
PV_dot_friction_3_list = np.full((np.size(traj['n_seeds']),int(duration / dt) + 1), np.nan)

PV_dot_heating_1_list = np.full((np.size(traj['n_seeds']),int(duration / dt) + 1), np.nan)
PV_dot_heating_2_list = np.full((np.size(traj['n_seeds']),int(duration / dt) + 1), np.nan)
PV_dot_heating_3_list = np.full((np.size(traj['n_seeds']),int(duration / dt) + 1), np.nan)

PV_dot_heating_SWR_list = np.full((np.size(traj['n_seeds']),int(duration / dt) + 1), np.nan)
PV_dot_heating_LWR_list = np.full((np.size(traj['n_seeds']),int(duration / dt) + 1), np.nan)
PV_dot_heating_SWRCS_list = np.full((np.size(traj['n_seeds']),int(duration / dt) + 1), np.nan)
PV_dot_heating_LWRCS_list = np.full((np.size(traj['n_seeds']), int(duration / dt) + 1), np.nan)

# n_seeds_pos = []

time = np.arange(0, PV_dot.shape[0])
level = PV_dot['level'].to_numpy()
lat = PV_dot['latitude'].to_numpy()
lon = PV_dot['longitude'].to_numpy()

if not test :
    for i in trange(np.size(traj['n_seeds'].to_numpy()), desc="Processing"):
        # print(i)
        PV_dot_list[i] = interp_traj(i, PV_dot.to_numpy(), lat, lon, level, time, traj, varlev='m')

        PV_dot_heating_interp =  interp_traj(i, PV_dot_heating.to_numpy(), lat, lon, level, time, traj, varlev='m' )
        PV_dot_heating_list[i] =PV_dot_heating_interp

        PV_dot_friction_interp =  interp_traj(i, PV_dot_friction.to_numpy(), lat, lon, level, time, traj, varlev='m')
        PV_dot_friction_list[i] =PV_dot_friction_interp

        PV_dot_friction_1_list[i] =  interp_traj(i, all_data["PV_dot_friction_1"].to_numpy(), lat, lon, level, time, traj, varlev='m')
        PV_dot_friction_2_list[i] =  interp_traj(i,  all_data["PV_dot_friction_2"].to_numpy(), lat, lon, level, time, traj, varlev='m')
        PV_dot_friction_3_list[i] = interp_traj(i,  all_data["PV_dot_friction_3"].to_numpy(), lat, lon, level, time, traj, varlev='m')

        PV_dot_heating_1_list[i] = interp_traj(i, all_data["PV_dot_heating_1"].to_numpy(), lat, lon, level, time, traj, varlev='m')
        PV_dot_heating_2_list[i] = interp_traj(i, all_data["PV_dot_heating_2"].to_numpy(), lat, lon, level, time, traj, varlev='m')
        PV_dot_heating_3_list[i] = interp_traj(i, all_data["PV_dot_heating_3"].to_numpy(), lat, lon, level, time, traj, varlev='m')

        PV_dot_heating_SWR_interp = interp_traj(i, all_data["PV_dot_heating_SWR"].to_numpy(), lat, lon, level, time, traj, varlev='m')
        PV_dot_heating_SWR_list[i] = PV_dot_heating_SWR_interp
        PV_dot_heating_LWR_interp = interp_traj(i, all_data["PV_dot_heating_LWR"].to_numpy(), lat, lon, level, time, traj, varlev='m')
        PV_dot_heating_LWR_list[i] = PV_dot_heating_LWR_interp
        PV_dot_heating_SWRCS_list[i] = interp_traj(i, all_data["PV_dot_heating_SWRCS"].to_numpy(), lat, lon, level, time, traj, varlev='m')
        PV_dot_heating_LWRCS_list[i] = interp_traj(i, all_data["PV_dot_heating_LWRCS"].to_numpy(), lat, lon, level, time, traj, varlev='m')



PV_dot_heating_rad_list = PV_dot_heating_SWR_list + PV_dot_heating_LWR_list


var_u, var_v= 'u', 'v'
u = all_data_pressure[var_u]  #
v = all_data_pressure[var_v]  #


def delta_PV_accumulated_all(PV_dot_traj, pv_traj, dt_traj = 0.5) :
    PV_accumulated =  np.zeros(np.shape(pv_traj))
    # PV_accumulated[:,-1] = pv_traj[:,-1].values

    for t in np.arange(1, np.size(pv_traj['time_ind'])) :
        PV_accumulated[:,-1-t] = PV_accumulated[:,-t] + PV_dot_traj[:,-t] * dt_traj * 3600 #PV_accumulated[-1] + np.trapz(PV_dot_traj[-1:-1-t:-1], x = time_list[-1:-1-t:-1].values * 3600)
       # print(str(t) + ' ' + str(PV_dot_traj[-1:-1-t:-1]))
    return PV_accumulated


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



# Levels
lev_U = 950 * 100




d0 = pd.to_datetime(n_date.data).day
h0 = pd.to_datetime(n_date.data).hour

from libs.traj import date_from_time_ind

time_plot = date_from_time_ind(time_ind, h0, d0)
print(time_plot)


landmask =mask['landseamask'].sel(latitude = slice(lat_min_3d, lat_max_3d))
landmask = xr.concat([ landmask.sel(longitude = slice(180, 360) ), landmask.sel(longitude = slice(0, 180))], dim = 'longitude')
landmask_lon = landmask['longitude']




landmask_lon= xr.where(landmask_lon > 180, landmask_lon - 360, landmask_lon, keep_attrs=True )


landmask['longitude'] = landmask_lon

# print(all_data_pressure.longitude.values)
# print(landmask['longitude'].values)
landmask = landmask.sel(longitude = slice(lon_min_3d, lon_max_3d))


lon2d = landmask["longitude"][:]  # (lat, lon)
lat2d = landmask["latitude"][:]  # (lat, lon)




points = landmask.where((landmask > 24.5) & (landmask < 25.5), 1, np.nan)
lon_mask = points['longitude'].values
lat_mask = points['latitude'].values
Lon_mask, Lat_mask = np.meshgrid(lon_mask, lat_mask)

# plt.show()

# 
# plt.figure()
# landmask.plot.contour(levels = [25] , colors = 'black') # Plot contour curves
# plt.show()


lon = u.sel(longitude = slice(lon_min_3d, lon_max_3d), latitude = slice(lat_max_3d, lat_min_3d) ).longitude.values
lat = u.sel(longitude = slice(lon_min_3d, lon_max_3d), latitude = slice(lat_max_3d, lat_min_3d) ).latitude.values
Lon, Lat = np.meshgrid(lon, lat)



vmin, vmax = 0, 30 # float(da.min()), float(da.max())

lev_min = 850
lev_max = 950

traj_lat = traj['lat'].isel(time_ind=0)
traj_lon = traj['lon'].isel(time_ind=0)
traj_P = traj['P'].isel(time_ind=0)


uo_sampled = u.sel(time=time_plot,level=lev_U , method='nearest').sel(longitude = slice(lon_min_3d, lon_max_3d), latitude = slice(lat_max_3d, lat_min_3d) )
vo_sampled = v.sel(time=time_plot, level=lev_U, method='nearest').sel(longitude = slice(lon_min_3d, lon_max_3d), latitude = slice(lat_max_3d, lat_min_3d) )


weights = np.cos(np.deg2rad(uo_sampled.latitude))
weights_matrix = np.tile(weights, (len(uo_sampled.longitude), 1)).T  # Adjust shape to match data


uo_weighted = uo_sampled * weights_matrix
vo_weighted = vo_sampled * weights_matrix


U = np.sqrt(uo_sampled ** 2 + vo_sampled ** 2)
Z = np.full_like(Lon, lev_U / 100)


fig = plt.figure(figsize=([6.4 * 2 , 4.8 * 4 ]))


ax3 = fig.add_subplot(423, projection="3d")


surf = ax3.plot_surface(
Lon, Lat, Z, #levels = np.arange(276,294,3),
facecolors=plt.cm.coolwarm((U - vmin) / (vmax - vmin)),
    # rstride=1, cstride=5,
     antialiased=True,
    #linewidth=1,
    shade=False, alpha=0.1
)



scat = ax3.scatter(traj_lon,
           traj_lat,
                  traj_P / 100 +1, c= 1e6 *  delta_PV_accumulated_all(PV_dot_heating_rad_list, traj['pv'])[:,time_ind] ,      depthshade=True , cmap = 'PiYG', vmin = -2.5, vmax = 2.5) #



landmask.plot.contour(ax = ax3, levels = [25], colors = 'black', offset = 949.9)  # Plot contour curves


ax3.set_zlim(lev_max, lev_min)



ax3.set_xlabel("Longitude")
ax3.set_ylabel("Latitude")
ax3.set_zlabel("Pressure (hPa)")


ax3.annotate(
    r"$\Delta PV_{rad}$",
    xy=(1.2, 0.5), xycoords='axes fraction',
    xytext=(+0.5, -0.5), textcoords='offset fontsize',
    fontsize=15,
    verticalalignment='top', fontfamily='serif',)


ax5 = fig.add_subplot(425, projection="3d")

surf = ax5.plot_surface(
Lon, Lat, Z,
facecolors=plt.cm.coolwarm((U - vmin) / (vmax - vmin)),
     antialiased=True,
    shade=False, alpha=0.1
)


scat = ax5.scatter(traj_lon,
           traj_lat,
                  traj_P / 100 +1, c= 1e6 *  delta_PV_accumulated_all(PV_dot_heating_list - PV_dot_heating_rad_list , traj['pv'])[:,time_ind] ,      depthshade=True , cmap = 'PiYG', vmin = -2.5, vmax = 2.5) #


landmask.plot.contour(ax = ax5, levels = [25], colors = 'black', offset = 949.9)  # Plot contour curves
ax5.set_zlim(lev_max, lev_min)


ax5.set_xlabel("Longitude")
ax5.set_ylabel("Latitude")
ax5.set_zlabel("Pressure (hPa)")


ax5.annotate(
    r"$\Delta PV_{heating - rad}$",
    xy=(1.1, 0.5), xycoords='axes fraction',
    xytext=(+0.5, -0.5), textcoords='offset fontsize',
    fontsize=15,
    verticalalignment='top', fontfamily='serif',)



ax1 = fig.add_subplot(421, projection="3d")





surf = ax1.plot_surface(
Lon, Lat, Z,
facecolors=plt.cm.coolwarm((U - vmin) / (vmax - vmin)),
     antialiased=True,
    shade=False, alpha=0.1
)


scat = ax1.scatter(traj_lon,
           traj_lat,
                  traj_P / 100 +1, c= 1e6 *  delta_PV_accumulated_all(PV_dot_friction_list , traj['pv'])[:,time_ind] ,      depthshade=True , cmap = 'PiYG', vmin = -2.5, vmax = 2.5) #


landmask.plot.contour(ax = ax1, levels = [25], colors = 'black', offset = 949.9)  # Plot contour curves

ax1.set_zlim(lev_max, lev_min)



ax1.set_xlabel("Longitude")
ax1.set_ylabel("Latitude")
ax1.set_zlabel("Pressure (hPa)")


ax1.annotate(
    r"$\Delta PV_{friction}$",
    xy=(1.2, 0.5), xycoords='axes fraction',
    xytext=(+0.5, -0.5), textcoords='offset fontsize',
    fontsize=15,
    verticalalignment='top', fontfamily='serif',)


ax1.set_title(r"Baroclinic phase", fontsize = 20)

ax7 = fig.add_subplot(427, projection="3d")



traj_dPV = traj['pv'].isel(time_ind=time_ind)  - traj['pv'].isel(time_ind=-1)


# for lev in levels_to_plot:
scat = ax7.scatter(traj_lon,
           traj_lat,
                  traj_P / 100   ,c= 1e6 *   traj_dPV ,      depthshade = True , cmap = 'PiYG', vmin = -2.5, vmax = 2.5, ) #

ax7.set_zlim(lev_max, lev_min)
Z_mask = np.full_like(Lon_mask, lev_U / 100  )


surf = ax7.plot_surface(
Lon, Lat, Z,
facecolors=plt.cm.coolwarm((U - vmin) / (vmax - vmin)),
     antialiased=True,
    shade=False, alpha=0.1
)



landmask.plot.contour(ax = ax7, levels = [25], colors = 'black', offset = 949.9)  # Plot contour curves


ax7.set_xlabel("Longitude")
ax7.set_ylabel("Latitude")
ax7.set_zlabel("Pressure (hPa)")

ax7.annotate(
    r"$\Delta PV_{tot}$",
    xy=(1.2, 0.5), xycoords='axes fraction',
    xytext=(+0.5, -0.5), textcoords='offset fontsize',
    fontsize=15,
    verticalalignment='top', fontfamily='serif',)


cyc = "AC3B" # "AC3A"



if cyc == 'AC3A' :
    niter=4
    dt_traj=0.5
    Nhor= 48#20
    Np=21 #11
    dataset = "era5_forecasts_1608T14_-15E-33E_78N-66.0N_750-960hPa" #"era5_forecasts_1608T14_5E-25E_78N-73.0N_850-960hPa"

    days = "15a16"
    loc = '65N-86N'
    time_ini=  '2022-08-16T14'
    long_min = -15
    long_max = 35
    lat_min = 65
    lat_max = 82
    level_min=1000
    level_max=800
    ini_time_step = 19
    h0 =14 #9 #intial hour
    traj_duration  =19
    # txt_title = f"{region_name} - {cyc}"
    # txt_save = f"_{region_name.replace(' ', '_')}"

elif cyc =='AC3B' :
    niter=4
    dt_traj=0.5
    Nhor=40
    Np=21
    dataset = "era5_forecasts_1908T14_-10E-30E_89N-79.0N_750-960hPa"
    days = "17a19"
    loc = '74N-90N'
    time_ini= '2022-08-19T14'
    long_min = -20 #-20
    long_max = 60
    lat_min = 74 #65
    lat_max = 90

    level_min = 1000
    level_max = 800
    ini_time_step = 43 #31 #15 #7 #Initial time step of the trajectory 15 #
    h0 = 14 #intial hour
    lon_min_1 = -30
    lon_max_1 = 120
    lat_min_1 = 75
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





path_traj = f'{root_input}/Generated_data/data/era5_forecasts/Trajectories/{dataset}/Traj_time_step_{ini_time_step}_{traj_duration}h_dt{dt_traj}_niter{niter}_Nhor{Nhor}_Np{Np}.nc'
#path_traj = f'{root_input}/Generated_data/data/Trajectories/{dataset}/Traj_time_step_31_PM_model_levels.nc'
traj_ini = xr.open_dataset(path_traj)

dt = traj_ini['time'].isel(n_seeds=0, time_ind=0).data - traj_ini['time'].isel(n_seeds=0, time_ind=1).data
duration = traj_ini['time'].isel(n_seeds=0, time_ind=0).data - traj_ini['time'].isel(n_seeds=0, time_ind=-1).data
duration = 12
t_list = np.linspace(h0, h0 - duration, int(duration / dt) + 1, endpoint=True)


path_data = f'{root_input}/Generated_data/data/era5_forecasts/ERA5_forecasts_all_pv_variables_all_levels_2022-08-{days}_{loc}.nc'
path_data_pressure =  f'{root_input}/Generated_data/data/era5_forecasts/ERA5_forecasts_all_pv_variables_700-1000hPa_per10hPa_2022-08-{days}_{loc}.nc'

#%%

if cyc =='AC3B' : #in order to remove the point at 90°N (divergence in PV)
    all_data = xr.open_dataset(path_data).sel(longitude = slice(lon_min_1, lon_max_1)).sel(latitude = slice(lat_max_1, lat_min_1))
    all_data_pressure = xr.open_dataset(path_data_pressure).sel(longitude = slice(lon_min_1, lon_max_1)).sel(latitude = slice(lat_max_1, lat_min_1)).rename({"plev":"level"})
    Sp  = xr.open_dataset(f'{root_input}/Data/era5_forecasts/ERA5_SP_2022-08-{days}_{loc}.nc').sel(longitude = slice(lon_min_1, lon_max_1)).sel(latitude = slice(lat_max_1, lat_min_1))
else :
    all_data = xr.open_dataset(path_data)
    all_data_pressure = xr.open_dataset(path_data_pressure).rename({"plev":"level"})
    Sp  = xr.open_dataset(f'{root_input}/Data/era5_forecasts/ERA5_SP_2022-08-{days}_{loc}.nc')


mask = xr.open_dataset("/media/chabranoo/LaCie/PhD/Data/IMERG_land_sea_mask.nc")
mask = mask.rename({'lon': 'longitude', 'lat': 'latitude'}) #.expand_dims({'level': all_data_pressure['level'] / 100})
#
#

all_data = all_data.sel(latitude=slice(None, None, -1))#.sel(level = slice(100000, 70000))

all_data_pressure = all_data_pressure.where(all_data_pressure['level'] < Sp['sp'], np.nan)

PV = all_data['PV_1'] + all_data['PV_2'] + all_data['PV_3']#.isel(time = slice(19, 32)).sel(latitude = slice(lat_max, lat_min), longitude = slice(long_min, long_max), level = slice(level_max, level_min))
PV_pressure = all_data_pressure['PV_1'] +  all_data_pressure['PV_2'] +   all_data_pressure['PV_3']
#%%
time = np.arange(0, np.size(PV['time']))
#%%


n_date = PV['time'].isel(time = int(traj_ini['time'].isel(n_seeds = 0,time_ind = 0)))
date = f'{pd.to_datetime(n_date.data).year}-{pd.to_datetime(n_date.data).month}-{pd.to_datetime(n_date.data).day}'
#%%


from libs.traj import is_in_region

n_list = is_in_region(traj_ini, cyc)

traj = traj_ini.sel(n_seeds = n_list).sel(time_ind = slice(0, int(duration  / dt + 1)))




#%%
#n_point = []

isnan_P = ~np.isnan(traj['P'])

n_point = np.sum(isnan_P, axis = 0)




mean_traj = traj.mean('n_seeds')



PV_dot_heating =all_data['PV_dot_heating_1'] + all_data['PV_dot_heating_2'] + all_data['PV_dot_heating_3']
PV_dot_friction =all_data['PV_dot_friction_1'] + all_data['PV_dot_friction_2'] + all_data['PV_dot_friction_3']
#%%
PV_dot = PV_dot_heating + PV_dot_friction

PV_dot_list = np.full((np.size(traj['n_seeds']),int(duration / dt) + 1), np.nan)
PV_dot_heating_list = np.full((np.size(traj['n_seeds']),int(duration / dt) + 1), np.nan)
PV_dot_friction_list = np.full((np.size(traj['n_seeds']),int(duration / dt) + 1), np.nan)

PV_dot_friction_1_list = np.full((np.size(traj['n_seeds']),int(duration / dt) + 1), np.nan)
PV_dot_friction_2_list = np.full((np.size(traj['n_seeds']),int(duration / dt) + 1), np.nan)
PV_dot_friction_3_list = np.full((np.size(traj['n_seeds']),int(duration / dt) + 1), np.nan)

PV_dot_heating_1_list = np.full((np.size(traj['n_seeds']),int(duration / dt) + 1), np.nan)
PV_dot_heating_2_list = np.full((np.size(traj['n_seeds']),int(duration / dt) + 1), np.nan)
PV_dot_heating_3_list = np.full((np.size(traj['n_seeds']),int(duration / dt) + 1), np.nan)

PV_dot_heating_SWR_list = np.full((np.size(traj['n_seeds']),int(duration / dt) + 1), np.nan)
PV_dot_heating_LWR_list = np.full((np.size(traj['n_seeds']),int(duration / dt) + 1), np.nan)
PV_dot_heating_SWRCS_list = np.full((np.size(traj['n_seeds']),int(duration / dt) + 1), np.nan)
PV_dot_heating_LWRCS_list = np.full((np.size(traj['n_seeds']), int(duration / dt) + 1), np.nan)

# n_seeds_pos = []

time = np.arange(0, PV_dot.shape[0])
level = PV_dot['level'].to_numpy()
lat = PV_dot['latitude'].to_numpy()
lon = PV_dot['longitude'].to_numpy()

if not test :
    for i in trange(np.size(traj['n_seeds'].to_numpy()), desc="Processing"):
        # print(i)
        PV_dot_list[i] = interp_traj(i, PV_dot.to_numpy(), lat, lon, level, time, traj, varlev='m')

        PV_dot_heating_interp =  interp_traj(i, PV_dot_heating.to_numpy(), lat, lon, level, time, traj, varlev='m' )
        PV_dot_heating_list[i] =PV_dot_heating_interp

        PV_dot_friction_interp =  interp_traj(i, PV_dot_friction.to_numpy(), lat, lon, level, time, traj, varlev='m')
        PV_dot_friction_list[i] =PV_dot_friction_interp

        PV_dot_friction_1_list[i] =  interp_traj(i, all_data["PV_dot_friction_1"].to_numpy(), lat, lon, level, time, traj, varlev='m')
        PV_dot_friction_2_list[i] =  interp_traj(i,  all_data["PV_dot_friction_2"].to_numpy(), lat, lon, level, time, traj, varlev='m')
        PV_dot_friction_3_list[i] = interp_traj(i,  all_data["PV_dot_friction_3"].to_numpy(), lat, lon, level, time, traj, varlev='m')

        PV_dot_heating_1_list[i] = interp_traj(i, all_data["PV_dot_heating_1"].to_numpy(), lat, lon, level, time, traj, varlev='m')
        PV_dot_heating_2_list[i] = interp_traj(i, all_data["PV_dot_heating_2"].to_numpy(), lat, lon, level, time, traj, varlev='m')
        PV_dot_heating_3_list[i] = interp_traj(i, all_data["PV_dot_heating_3"].to_numpy(), lat, lon, level, time, traj, varlev='m')

        PV_dot_heating_SWR_interp = interp_traj(i, all_data["PV_dot_heating_SWR"].to_numpy(), lat, lon, level, time, traj, varlev='m')
        PV_dot_heating_SWR_list[i] = PV_dot_heating_SWR_interp
        PV_dot_heating_LWR_interp = interp_traj(i, all_data["PV_dot_heating_LWR"].to_numpy(), lat, lon, level, time, traj, varlev='m')
        PV_dot_heating_LWR_list[i] = PV_dot_heating_LWR_interp
        PV_dot_heating_SWRCS_list[i] = interp_traj(i, all_data["PV_dot_heating_SWRCS"].to_numpy(), lat, lon, level, time, traj, varlev='m')
        PV_dot_heating_LWRCS_list[i] = interp_traj(i, all_data["PV_dot_heating_LWRCS"].to_numpy(), lat, lon, level, time, traj, varlev='m')



PV_dot_heating_rad_list = PV_dot_heating_SWR_list + PV_dot_heating_LWR_list




var_u, var_v= 'u', 'v'
u = all_data_pressure[var_u]  #
v = all_data_pressure[var_v]  #



if cyc == 'AC3A' :
    lon_min_3d = -30
    lon_max_3d = 40
    lat_max_3d = 80
    lat_min_3d = 65
elif cyc =='AC3B' :
    lon_min_3d = -30
    lon_max_3d = 40
    lat_max_3d = 90
    lat_min_3d = 75
elif cyc =='AC4' :
    lon_min_3d = 10
    lon_max_3d = 60
    lat_max_3d = 80
    lat_min_3d = 70



# Levels
lev_U = 950 * 100

d0 = pd.to_datetime(n_date.data).day
h0 = pd.to_datetime(n_date.data).hour



time_plot = date_from_time_ind(time_ind, h0, d0)
print(time_plot)

landmask =mask['landseamask'].sel(latitude = slice(lat_min_3d, lat_max_3d))
landmask = xr.concat([ landmask.sel(longitude = slice(180, 360) ), landmask.sel(longitude = slice(0, 180))], dim = 'longitude')
landmask_lon = landmask['longitude']




landmask_lon= xr.where(landmask_lon > 180, landmask_lon - 360, landmask_lon, keep_attrs=True )


landmask['longitude'] = landmask_lon

print(all_data_pressure.longitude.values)
print(landmask['longitude'].values)
landmask = landmask.sel(longitude = slice(lon_min_3d, lon_max_3d))


lon2d = landmask["longitude"][:]  # (lat, lon)
lat2d = landmask["latitude"][:]  # (lat, lon)



points = landmask.where((landmask > 24.5) & (landmask < 25.5), 1, np.nan)
lon_mask = points['longitude'].values
lat_mask = points['latitude'].values
Lon_mask, Lat_mask = np.meshgrid(lon_mask, lat_mask)


# Make lon/lat meshz
lon = u.sel(longitude = slice(lon_min_3d, lon_max_3d), latitude = slice(lat_max_3d, lat_min_3d) ).longitude.values
lat = u.sel(longitude = slice(lon_min_3d, lon_max_3d), latitude = slice(lat_max_3d, lat_min_3d) ).latitude.values
Lon, Lat = np.meshgrid(lon, lat)


# Global color limits for comparability
vmin, vmax = 0, 30 # float(da.min()), float(da.max())

lev_min = 850
lev_max = 950

traj_lat = traj['lat'].isel(time_ind=0)
traj_lon = traj['lon'].isel(time_ind=0)
traj_P = traj['P'].isel(time_ind=0)


uo_sampled = u.sel(time=time_plot, level=lev_U , method='nearest').sel(longitude = slice(lon_min_3d, lon_max_3d), latitude = slice(lat_max_3d, lat_min_3d) )
vo_sampled = v.sel(time=time_plot, level=lev_U, method='nearest').sel(longitude = slice(lon_min_3d, lon_max_3d), latitude = slice(lat_max_3d, lat_min_3d) )


weights = np.cos(np.deg2rad(uo_sampled.latitude))
weights_matrix = np.tile(weights, (len(uo_sampled.longitude), 1)).T  # Adjust shape to match data

uo_weighted = uo_sampled * weights_matrix
vo_weighted = vo_sampled * weights_matrix


U = np.sqrt(uo_sampled ** 2 + vo_sampled ** 2)

Z = np.full_like(Lon, lev_U / 100)





ax4 = fig.add_subplot(424, projection="3d")


surf = ax4.plot_surface(
Lon, Lat, Z, #levels = np.arange(276,294,3),
facecolors=plt.cm.coolwarm((U - vmin) / (vmax - vmin)),
    # rstride=1, cstride=5,
     antialiased=True,
    #linewidth=1,
    shade=False, alpha=0.1
)



scat = ax4.scatter(traj_lon,
           traj_lat,
                  traj_P / 100 +1, c= 1e6 *  delta_PV_accumulated_all(PV_dot_heating_rad_list, traj['pv'])[:,time_ind] ,      depthshade=True , cmap = 'PiYG', vmin = -2.5, vmax = 2.5) #



landmask.plot.contour(ax = ax4, levels = [25], colors = 'black', offset = 949.9)  # Plot contour curves






ax4.set_zlim(lev_max, lev_min)



ax4.set_xlabel("Longitude")
ax4.set_ylabel("Latitude")
ax4.set_zlabel("Pressure (hPa)")










ax6 = fig.add_subplot(426, projection="3d")






surf = ax6.plot_surface(
Lon, Lat, Z, #levels = np.arange(276,294,3),
facecolors=plt.cm.coolwarm((U - vmin) / (vmax - vmin)),
    # rstride=1, cstride=5,
     antialiased=True,
    #linewidth=1,
    shade=False, alpha=0.1
)



# for lev in levels_to_plot:
scat = ax6.scatter(traj_lon,
           traj_lat,
                  traj_P / 100 +1, c= 1e6 *  delta_PV_accumulated_all(PV_dot_heating_list - PV_dot_heating_rad_list , traj['pv'])[:,time_ind] ,      depthshade=True , cmap = 'PiYG', vmin = -2.5, vmax = 2.5) #


landmask.plot.contour(ax = ax6, levels = [25], colors = 'black', offset = 949.9)  # Plot contour curves
ax6.set_zlim(lev_max, lev_min)


ax6.set_xlabel("Longitude")
ax6.set_ylabel("Latitude")
ax6.set_zlabel("Pressure (hPa)")




ax2 = fig.add_subplot(422, projection="3d")





surf = ax2.plot_surface(
Lon, Lat, Z, #levels = np.arange(276,294,3),
facecolors=plt.cm.coolwarm((U - vmin) / (vmax - vmin)),
    # rstride=1, cstride=5,
     antialiased=True,
    #linewidth=1,
    shade=False, alpha=0.1
)


scat = ax2.scatter(traj_lon,
           traj_lat,
                  traj_P / 100 +1, c= 1e6 *  delta_PV_accumulated_all(PV_dot_friction_list , traj['pv'])[:,time_ind] ,      depthshade=True , cmap = 'PiYG', vmin = -2.5, vmax = 2.5) #

landmask.plot.contour(ax = ax2, levels = [25], colors = 'black', offset = 949.9)  # Plot contour curves

ax2.set_zlim(lev_max, lev_min)



ax2.set_xlabel("Longitude")
ax2.set_ylabel("Latitude")
ax2.set_zlabel("Pressure (hPa)")





#%%
# Levels

ax2.set_title(r"Cold-core phase" #"Cold core phase"
              , fontsize = 20)

ax8 = fig.add_subplot(428, projection="3d")








traj_dPV = traj['pv'].isel(time_ind=time_ind)  - traj['pv'].isel(time_ind=-1)


# for lev in levels_to_plot:
scat = ax8.scatter(traj_lon,
           traj_lat,
                  traj_P / 100   ,c= 1e6 *   traj_dPV ,      depthshade = True , cmap = 'PiYG', vmin = -2.5, vmax = 2.5, ) #



ax8.set_zlim(lev_max, lev_min)


Z_mask = np.full_like(Lon_mask, lev_U / 100  )



surf = ax8.plot_surface(
Lon, Lat, Z, #levels = np.arange(276,294,3),
facecolors=plt.cm.coolwarm((U - vmin) / (vmax - vmin)),
    # rstride=1, cstride=5,
     antialiased=True,
    #linewidth=1,
    shade=False, alpha=0.1
)



landmask.plot.contour(ax = ax8, levels = [25], colors = 'black', offset = 949.9)  # Plot contour curves


ax8.set_xlabel("Longitude")
ax8.set_ylabel("Latitude")
ax8.set_zlabel("Pressure (hPa)")






fig.subplots_adjust(bottom=0.1, wspace = 0.2, hspace = 0.1)
cbar_ax = fig.add_axes([0.2, 0.07, 0.3,0.01])

# cbar_ax = fig.add_axes([0, 0.05, 1, 0.02])
cbar_ax2 = fig.add_axes([0.6, 0.07, 0.3, 0.01])


label_list = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)']
axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]
for i in range(len(label_list)):
    axi = axs[i]
    axi.annotate(
        label_list[i],
        xy=(0, 1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize='large', verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='none', edgecolor='none', pad=3.0))


# Colorbar
mappable = plt.cm.ScalarMappable(cmap="coolwarm")
mappable.set_array([vmin, vmax])
cbar = fig.colorbar(mappable, cax=cbar_ax,orientation='horizontal')#, fraction=0.1, pad=0.2,)
cbar.set_label(r"U [m.s$^{-1}$]")

cbar_scat = fig.colorbar(scat, cax=cbar_ax2,orientation='horizontal')
cbar_scat.set_label(r"$\Delta PV$ [pvu]")


plt.savefig(f"{path_plots}/delta_pv_3D_synthesis.png", bbox_inches='tight', transparent = True)
plt.show()