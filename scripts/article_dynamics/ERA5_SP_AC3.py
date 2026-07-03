from matplotlib.lines import lineStyles
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
from matplotlib.transforms import ScaledTranslation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.ma import masked_array


dataset = 'ERA5'

cyc = "AC3" # "AC3A"


local = True


time_ini=  '2022-08-16T14'
long_min = -40
long_max = 100
lat_min = 62
lat_max = 100
level_min=400
level_max=1050





root_input = "/data/nchiab/PV"

if local :
    root_input = "/media/chabranoo/LaCie/PhD"



path_plots = f"{root_input}/Generated_data/plots/ERA5/"


def circle(lat,lon,R) :
    R_earth = 6371.0  # km
    lat0 = np.radians(lat)
    lon0 = np.radians(lon)
    delta = R / R_earth

    lat_circle = np.arcsin(np.sin(lat0) * np.cos(delta) + np.cos(lat0) * np.sin(delta) * np.cos(np.linspace(0, 2 * np.pi, 100)))
    lon_circle = lon0 + np.arctan2(np.sin(np.linspace(0, 2 * np.pi, 100)) * np.sin(delta) * np.cos(lat0), np.cos(delta) - np.sin(lat0) * np.sin(lat_circle))

    return np.degrees(lat_circle), np.degrees(lon_circle)



#%%

#var_pv = 'pv'
#path_PV =f'{root_input}/Generated_data/data/era5_forecasts/PV/ERA5_forecasts_u_v_w_temp_PV_pres_all_levels_2022-08-{days}_{loc}.nc'
#path_PV =f'{root_input}/Generated_data/data/PV/ERA5_forecasts/ERA5_U_V_W_TEMP_PV_PM_all_levels_2022-08-15a16_50N-90N.nc'
path_data_ta = f'{root_input}/Data/ERA5/ta.202208.ap1e5.GLOBAL_025.nc'
path_data_ci=  f'{root_input}/Data/ERA5/ci.202208.as1e5.GLOBAL_025.nc'
path_data_msl=  f'{root_input}/Data/ERA5/msl.202208.as1e5.GLOBAL_025.nc'
path_data_sp = f'{root_input}/Data/ERA5/sp.202208.as1e5.GLOBAL_025.nc'
# path_traj = f'{root_input}/Data/ERA5/AC3_Nicolas.csv'
path_traj = f'{root_input}/Data/ERA5/traj_AC3A.csv'
ta = xr.open_dataset(path_data_ta)['ta'].sel(level = slice(level_min, level_max)).sel(latitude = slice(lat_max, 50))
ci = xr.open_dataset(path_data_ci)['siconc'].sel(latitude = slice(lat_max, 50))
msl  = xr.open_dataset(path_data_msl)['msl'].sel(latitude = slice(lat_max, 50))
# sp  = xr.open_dataset(path_data_sp)['sp'].sel(time = slice('2022-08-14', '2022-08-22'))


path_flight_F46 = "/bdd/RALI-ThinIce/Data/RASTA/L2-wind-product_06.3.1/THINICE_20220816_F46_RASTA_WIND_final_06.3.1.nc"
path_flight_F49 = f'/bdd/RALI-ThinIce/Data/RASTA/L2-wind-product_06.3.1/THINICE_20220819_F49_RASTA_WIND_final_06.3.1.nc'


if local:
    path_flight_F46 = f'/media/chabranoo/LaCie/PhD/data/rali/THINICE_F46_RASTA_WIND_final_06.3.1.nc'
    path_flight_F49 = f'/media/chabranoo/LaCie/PhD/data/rali/THINICE_F49_RASTA_WIND_final_06.3.1.nc'

flight_F46 = xr.open_dataset(path_flight_F46)
flight_F49 = xr.open_dataset(path_flight_F49)

R = 1000  # radius in km
R_center = 100
delta_r = 50


traj_csv =  pd.read_csv(path_traj)#, delimiter=';', header = True, names = header)
traj = traj_csv.set_index('time').to_xarray()
traj['time'] = pd.to_datetime(traj['time'], format='%Y-%m-%d %H:%M:%S')
traj = traj.sortby('time')
traj['msl'] = traj['msl'] / 100  # convert to hPa

def theta_f(T, P, P0 = 1000) :
    return T * (P0 / P)**0.286


# haversine distance formula
def haversine(lat1, lon1, lat2, lon2):
    R_earth = 6371.0  # km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R_earth * np.arcsin(np.sqrt(a))


lat0 = traj['lat'].sel(time='2022-08-16T14', method = 'nearest').data
lon0 = traj['lon'].sel(time='2022-08-16T14', method = 'nearest').data

dist = xr.apply_ufunc(
    haversine,
    lat0,
    lon0,
    ta.latitude,
    ta.longitude,
)

# select only points inside the radius

lat_circle, lon_circle = circle(lat0, lon0, R)
# circle = dist.where(dist <= R + delta_r)
# circle = circle.where(dist >= R - delta_r)
# circle = xr.where(~np.isnan(circle),1,np.nan, keep_attrs=True )

# print(circle)

# fig = plt.figure()#figsize=(10, 10))
# #proj = ccrs.NorthPolarStereo()  # Change projection as needed
#
#
# proj = ccrs.NorthPolarStereo()  # Change projection as needed
# ax = plt.axes(projection=proj)
#
# # Plot coastlines and set the global extent
# ax.coastlines()
# ax.set_global()
# ax.set_extent([long_min, long_max, lat_min, lat_max], ccrs.PlateCarree())
# ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
#
# # Plot the unmasked topo data
#
#
# # Add other data variables
# im = ta.sel(time = '2022-08-16T14') .sel(level = 950).plot.contourf(ax=ax, add_colorbar=False, levels = np.arange(260, 310, 4), cmap = 'coolwarm', transform=ccrs.PlateCarree() )
# # circle.plot.contour(ax=ax, add_colorbar=False, levels = [0,1.5], color = 'black', transform=ccrs.PlateCarree() )
# cbar = fig.colorbar(im, ax=ax)#, orientation='vertical', fraction=0.046, pad=0.04)
# ax.plot(lon_circle, lat_circle, transform=ccrs.PlateCarree( ), color = 'black', label = '1,000 km radius')
# # ax.plot(traj['lon'], traj['lat'], transform=ccrs.Geodetic(), color = 'red')
# ax.scatter(traj['lon'].sel(time = '2022-08-16T14', method = 'nearest'), traj['lat'].sel(time = '2022-08-16T14', method = 'nearest'), transform=ccrs.Geodetic(), color = 'blue', label = 'AC3A')
#
# ax.legend()
#
#
#
# # cbar.set_label(r'$\dot{\theta}$ [K / s]')
#
# lat0 = traj['lat'].sel(time='2022-08-19T14', method = 'nearest').data
# lon0 = traj['lon'].sel(time='2022-08-19T14', method = 'nearest').data
# lat_circle, lon_circle = circle(lat0, lon0, R)
#
# # plt.savefig(f"{path_plots}/rad_heating_dominant/theta_dot_lwr_{pres}hPa_{time_cs}_region{region}_{dataset}.png", bbox_inches='tight', transparent=True)
# plt.show()
#
# fig = plt.figure()#figsize=(10, 10))
# #proj = ccrs.NorthPolarStereo()  # Change projection as needed
#
#
# proj = ccrs.NorthPolarStereo()  # Change projection as needed
# ax = plt.axes(projection=proj)
#
# # Plot coastlines and set the global extent
# ax.coastlines()
# ax.set_global()
# ax.set_extent([long_min, long_max, lat_min, lat_max], ccrs.PlateCarree())
# ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
#
# # Plot the unmasked topo data
#
#
# # Add other data variables
# im = ta.sel(time = '2022-08-19T14').sel(level = 950).plot.contourf(ax=ax, add_colorbar=False, levels = 10, cmap = 'coolwarm', transform=ccrs.PlateCarree() )
# cbar = fig.colorbar(im, ax=ax)#, orientation='vertical', fraction=0.046, pad=0.04)
# ax.plot(lon_circle, lat_circle, transform=ccrs.PlateCarree( ), color = 'black', label = '1,000 km radius')
# # ax.plot(traj['lon'], traj['lat'], transform=ccrs.Geodetic(), color = 'red')
# # ax.scatter(traj['lon'].sel(time = '2022-08-16T14', method = 'nearest'), traj['lat'].sel(time = '2022-08-16T14', method = 'nearest'), transform=ccrs.Geodetic(), color = 'blue', label = 'AC3A')
# ax.scatter(traj['lon'].sel(time = '2022-08-19T14', method = 'nearest'), traj['lat'].sel(time = '2022-08-19T14', method = 'nearest'), transform=ccrs.Geodetic(), color = 'green', label = 'Cold core')
# ax.legend()
#
#
#
# # cbar.set_label(r'$\dot{\theta}$ [K / s]')
#
#
# # plt.savefig(f"{path_plots}/rad_heating_dominant/theta_dot_lwr_{pres}hPa_{time_cs}_region{region}_{dataset}.png", bbox_inches='tight', transparent=True)
# plt.show()

lat0 = traj['lat'].isel(time =50)
lon0 = traj['lon'].isel(time =50)






# select only points inside the radius

circle = ta.isel(time =50).where(dist <= R + delta_r)
circle = circle.where(dist >= R - delta_r)


#
# # Create a figure and axis with your specified projection
# fig = plt.figure()#figsize=(10, 10))
# #proj = ccrs.NorthPolarStereo()  # Change projection as needed
#
#
# proj = ccrs.NorthPolarStereo()  # Change projection as needed
# ax = plt.axes(projection=proj)
#
# # Plot coastlines and set the global extent
# ax.coastlines()
# ax.set_global()
# ax.set_extent([long_min, long_max, lat_min, lat_max], ccrs.PlateCarree())
# ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
#
# # Plot the unmasked topo data
#
#
# # Add other data variables
# im = circle.sel(level = 850).plot.contourf(ax=ax, add_colorbar=False, levels = 10,  transform=ccrs.PlateCarree() )
# cbar = fig.colorbar(im, ax=ax)#, orientation='vertical', fraction=0.046, pad=0.04)
#
#
#
# # cbar.set_label(r'$\dot{\theta}$ [K / s]')
#
#
# # plt.savefig(f"{path_plots}/rad_heating_dominant/theta_dot_lwr_{pres}hPa_{time_cs}_region{region}_{dataset}.png", bbox_inches='tight', transparent=True)
# plt.show()

# ta_height_traj = np.zeros(( len(ta['level']), len(traj['time'])))
sp_list = np.zeros(len(traj['time']))
# theta_height_traj = np.zeros(( len(ta['level']), len(traj['time'])))

ta1_height_traj = np.zeros(( len(ta['level']), len(traj['time'])))
# sp_list = np.zeros(len(traj['time']))

# #
# for i in trange(len(traj['time'])) :
#     # ta_height_traj[:, i] = ta.sel(time = traj['time'][i]).interp(latitude = traj['lat'][i], longitude = traj['lon'][i]).to_numpy()
#     sp_list[i] = msl.sel(time=traj['time'][i]).interp(latitude=traj['lat'][i], longitude=traj['lon'][i]).to_numpy()
#     theta_height_traj[:, i] = theta.sel(time = traj['time'][i]).interp(latitude = traj['lat'][i], longitude = traj['lon'][i]).to_numpy()
if True :
    for i in trange(len(traj['time'])) :
        lat0 = traj['lat'].isel(time=i)
        lon0 = traj['lon'].isel(time=i)
        dist = xr.apply_ufunc(
            haversine,
            lat0,
            lon0,
            ta.latitude,
            ta.longitude,
        )

        # select only points inside the radius

        ta_mean = ta.sel(time=traj['time'][i]).where(dist <= R + delta_r)
        ta_mean = ta_mean.where(dist >= R - delta_r).mean('latitude').mean('longitude')



        dist = xr.apply_ufunc(
            haversine,
            lat0,
            lon0,
            ta.latitude,
            ta.longitude,
        )


        ta_center = ta.sel(time=traj['time'][i]).where(dist <= R_center ).mean('latitude').mean('longitude')


        ta1_height_traj[:, i] = (ta_center - ta_mean).to_numpy()





temp_height_time = xr.DataArray(
    data=ta1_height_traj,
    dims=[ "level", "time"],
    coords=dict(
        level=ta['level'],
        time=traj['time'],

    ),
    attrs=dict(
        description="Temperature anomaly",
        units="K",
    ),
)




# theta = theta_f(ta, ta['level'])



 # Cross sections
#12.4242

# Create a figure and axis with your specified projection
fig = plt.figure(figsize = (6.4, 10))#figsize=(10, 10))
#proj = ccrs.NorthPolarStereo()  # Change projection as needed


proj = ccrs.NorthPolarStereo()  # Change projection as needed
ax = plt.subplot(2,1,1,projection=proj)

# Plot coastlines and set the global extent
ax.coastlines()
ax.set_global()
ax.set_extent([long_min, long_max, lat_min, lat_max], ccrs.PlateCarree())
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

# Plot the unmasked topo data


# Add other data variables
im = ci.mean('time').plot.contourf(ax=ax, add_colorbar=False, levels = np.arange(0,1.01,0.2), cmap = 'Blues_r', transform=ccrs.PlateCarree() )
cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.036, pad=0.1,  shrink = 0.8)
#
#
# cbar_ax = ax.inset_axes([1.1, 0, 0.02, 0.5])
# cbar = fig.colorbar(im, cax=cbar_ax, orientation  = 'vertical')#, fraction=0.1, pad=0.2, orientation='horizontal')#,label = 'Potential vorticity [pvu]')#, fontsize = 20)
cbar.set_label(r'Sea ice fraction')

# ax.plot(traj['lon'], traj['lat'], transform=ccrs.Geodetic(), color = 'yellow')

# ax.scatter(traj['lon'][::6], traj['lat'][::6] ,  transform=ccrs.Geodetic(), c = 'black', marker = 'x')

# traj_pres = ax.scatter(traj['lon'], traj['lat'], c = traj['msl'] ,  transform=ccrs.Geodetic(), marker = 'o')#, facecolors="none", edgecolors='yellow',)
# ax.scatter(traj['lon'][::6], traj['lat'][::6] ,  transform=ccrs.Geodetic(),facecolors='none', edgecolors='black', marker = 'o')

#
# cbar_ax_pres = ax.inset_axes([1.1, 0.5, 0.02, 0.5])
# cbar_pres = fig.colorbar(traj_pres, cax=cbar_ax_pres, orientation  = 'vertical')#, fraction=0.1, pad=0.2, orientation='horizontal')#,label = 'Potential vorticity [pvu]')#, fontsize = 20)
# cbar_pres.set_label(r'Cyclone centre surface pressure [hPa]')

# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=1, projection = ccrs.PlateCarree())
# cba = plt.colorbar(im, cax=cax)
#
# cax = divider.append_axes("right", size="5%", pad=1.0, projection = ccrs.PlateCarree())
# cbb = plt.colorbar(traj_pres, cax=cax)

ax.plot(traj['lon'], traj['lat'],  transform=ccrs.Geodetic(), color = 'black') #,  linewidth = 3)
ax.scatter(traj['lon'][::6], traj['lat'][::6], s = (-traj['msl'][::6] + np.max(traj['msl'])) * 2 + 3,  transform=ccrs.Geodetic(),facecolors='none', edgecolors='black', marker = 'o', linewidth = 3, zorder = 9)
scatter = ax.scatter(traj['lon'][::2], traj['lat'][::2], s = ( -traj['msl'][::2] + np.max(traj['msl'])).values * 2,  transform=ccrs.Geodetic(), marker = 'o', facecolors="none", edgecolors='yellow', zorder = 10)
traj_min = traj.where(traj['msl'] == np.nanmin(traj['msl']), drop = True)

ax.scatter(traj['lon'][-1], traj['lat'][-1], s = (-traj['msl'][-1] + np.max(traj['msl'])) * 2 + 3,  transform=ccrs.Geodetic(),facecolors='none', edgecolors='green', marker = 'o', zorder = 11)
ax.scatter(traj['lon'][0], traj['lat'][0], s = (-traj['msl'][0] + np.max(traj['msl'])) * 2 + 3,  transform=ccrs.Geodetic(),facecolors='none', edgecolors='m', marker = 'o', zorder = 11)
# ax.scatter(traj['lon'][0], traj['lat'][0], transform=ccrs.Geodetic(), color='purple', marker = 'x', zorder = 9)
# ax.scatter(traj['lon'][-1], traj['lat'][-1], transform=ccrs.Geodetic(), color='green', marker = 'x', zorder = 9)

# ax.scatter(traj['lon'], traj['lat'], s =( -traj['msl'] + np.max(traj['msl'])) * 2 + 3,  transform=ccrs.Geodetic(), marker = 'o', facecolors="none", edgecolors='yellow',)

ax.scatter(traj.where(traj['msl'] == np.min(traj['msl'])  )['lon'], traj.where(traj['msl'] == np.min(traj['msl'])  )['lat'],
           s =( -traj.where(traj['msl'] == np.min(traj['msl']))['msl'] + np.max(traj['msl'])) * 2 + 3,  transform=ccrs.Geodetic(), marker = 'o', facecolors="none", edgecolors='orange', zorder = 12)

traj_second = traj.sel(time = slice('2022-08-21', None))
ax.scatter(traj_second.where(traj_second['msl'] == np.min(traj_second['msl'])  )['lon'], traj_second.where(traj_second['msl'] == np.min(traj_second['msl']))['lat'],
           s =( -traj_second.where(traj_second['msl'] == np.min(traj_second['msl']))['msl'] + np.max(traj['msl'])) * 2 + 3,  transform=ccrs.Geodetic(), marker = 'o',
           facecolors="None", edgecolors='orange',)
# print(flight_F46)
f46_plot, = ax.plot(flight_F46['longitude'], flight_F46['latitude'], c = 'cyan', transform=ccrs.PlateCarree(), label = 'F46')
f49_plot, = ax.plot(flight_F49['longitude'], flight_F49['latitude'], c = 'm', transform=ccrs.PlateCarree(), label = 'F49')
# legend0 = ax.legend(handles = [f46_plot, f49_plot], label = ['F46', 'F49'], loc = 'right')
# ax.add_artist(legend0)
s = 100
scat1 = ax.scatter(traj['lon'].sel(time = '2022-08-16T14', method = 'nearest'), traj['lat'].sel(time = '2022-08-16T14', method = 'nearest'), s = (-traj['msl'].sel(time = '2022-08-16T14', method = 'nearest') + np.max(traj['msl'])) * 2 + 3,marker = 'o', transform=ccrs.Geodetic(),
           facecolors='none', edgecolors='red', label = 'Neutral core', zorder = 50)
scat2 = ax.scatter(traj['lon'].sel(time = '2022-08-19T14', method = 'nearest'), traj['lat'].sel(time = '2022-08-19T14', method = 'nearest'), s = (-traj['msl'].sel(time = '2022-08-19T14', method = 'nearest') + np.max(traj['msl'])) * 2 + 3, marker = 'o', transform=ccrs.Geodetic(),
           facecolors='none', edgecolors='blue', label = 'Cold core', zorder = 50)
legend1 = ax.legend(handles = [scat1, scat2, f46_plot, f49_plot], loc = 'upper right')
ax.add_artist(legend1)
# handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
# legend2 = ax.legend(handles, labels, loc="lower right", title="Sizes")

kw = dict(prop = "sizes", num = 3, markerfacecolor="none", markeredgecolor='yellow', fmt="{x:.0f} hPa",
          func=lambda s:  -(s - 3) / 2 + np.max(traj['msl']).values)
legend2 = ax.legend(*scatter.legend_elements(**kw),
                    loc="lower right", title="Sea level pressure")

ax.annotate(
    'a)',
    xy=(0, 1), xycoords='axes fraction',
    xytext=(+0.5, -0.5), textcoords='offset fontsize',
    fontsize='medium', verticalalignment='top', fontfamily='serif',
    bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))



# cbar.set_label(r'$\dot{\theta}$ [K / s]')


# plt.savefig(f"{path_plots}/ci_mean_traj_AC3.png", bbox_inches='tight', transparent=True)
# plt.show()



ax1 = plt.subplot(4,1,3)
traj['msl'].plot(ax = ax1)
# plt.plot(traj['time'], sp_list, label = 'sp interp', linestyle = 'dashed')
ax1.vlines(['2022-08-16T14'], ymin = traj['msl'].min(), ymax = traj['msl'].max(), colors = 'red', linestyles = 'dashed', label = 'Neutral core')
ax1.vlines(['2022-08-19T14'], ymin = traj['msl'].min(), ymax = traj['msl'].max(), colors = 'blue', linestyles = 'dashed', label = 'Cold core')
ax1.legend()

ax1.grid()


# ax1.invert_yaxis()
ax1.set_ylabel('Sea level pressure [hPa]')
ax1.label_outer()
ax1.annotate(
    'b)',
    xy=(0, 1), xycoords='axes fraction',
    xytext=(+0.5, -0.5), textcoords='offset fontsize',
    fontsize='medium', verticalalignment='top', fontfamily='serif',
    bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))


ax2 = plt.subplot(4,1,4, sharex = ax1)


ta1_im = temp_height_time.plot.contourf(ax = ax2, levels = np.arange(-12, 12.2, 2), extend = 'both', cmap = 'coolwarm', add_colorbar = False)
# ta1_im = ax.contourf(traj['time'], ta['level'],  ta1_height_traj, levels = np.arange(-8, 8.2, 1), extend = 'both', cmap = 'coolwarm', colorbar = False)
# ax2.vlines(traj['time'].sel(time = '2022-08-16T14', method = 'nearest'), ymin = level_min, ymax = 1000, colors = 'red', linestyles = 'dashed', label = 'Neutral core')
# ax2.vlines(traj['time'].sel(time = '2022-08-19T14', method = 'nearest'), ymin = level_min, ymax = 1000, colors = 'blue', linestyles = 'dashed', label = 'Cold core')
ax2.vlines(['2022-08-16T14'], ymin = level_min, ymax = 1000, colors = 'red', linestyles = 'dashed',)
ax2.vlines(['2022-08-19T14'], ymin = level_min, ymax = 1000, colors = 'blue', linestyles = 'dashed', )
ax2.invert_yaxis()
# ax2.set_xlabel('August 2022')
ax2.set_xlabel('')
# cbar = fig.colorbar(ta1_im, orientation='horizontal', fraction=0.1, pad=0.2, ax = ax2)

# juste afficher les jours en xthicks

# x_list = []
# labels = []
# for i in range(len(traj['time'])) :
#     if int(traj['time'][i].dt.hour.values) == 0 :
#         x_list.append(int(traj['time_ind'][i].values))
#         labels.append(int(traj['time'][i].dt.day.values))
# ax2.set_xticks(x_list, labels)



ax2.set_ylabel('[hPa]')
# plt.savefig(f"{path_plots}/ta_center_minus_tmean_AC3.png", bbox_inches='tight', transparent=True)
ax2.grid()

ax2.annotate(
    'c)',
    xy=(0, 1), xycoords='axes fraction',
    xytext=(+0.5, -0.5), textcoords='offset fontsize',
    fontsize='medium', verticalalignment='top', fontfamily='serif',
    bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))


fig.subplots_adjust(hspace=0.07, bottom=0.1)

cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.02])
cbar = fig.colorbar(ta1_im, cax=cbar_ax, fraction=0.1, pad=0.2, orientation='horizontal')#,label = 'Potential vorticity [pvu]')#, fontsize = 20)
cbar.set_label(r'Temperature anomaly [K]')
# cbar.ax.tick_params(labelsize=20)

plt.savefig(f"{path_plots}/traj_temp_sp_AC3.png", bbox_inches='tight', transparent=True)
plt.show()







# fig = plt.figure()
# temp_im = plt.contourf(traj['time_ind'], ta['level'],  ta_height_traj, levels = 20, cmap = 'coolwarm')
# plt.vlines(traj['time_ind'].sel(time = '2022-08-16T14', method = 'nearest'), ymin = level_min, ymax = 1000, colors = 'black', linestyles = 'dashed', label = 'Neutral core')
# plt.vlines(traj['time_ind'].sel(time = '2022-08-19T14', method = 'nearest'), ymin = level_min, ymax = 1000, colors = 'black', linestyles = 'dashed', label = 'Cold core')
# plt.gca().invert_yaxis()
#
# cbar = fig.colorbar(temp_im, orientation='vertical', fraction=0.046, pad=0.04)
# cbar.set_label(r'Temperature[K]')
# # juste afficher les jours en xthicks
# x_list = []
# labels = []
# for i in range(len(traj['time'])) :
#     if int(traj['time'][i].dt.hour.values) == 0 :
#         x_list.append(int(traj['time_ind'][i].values))
#         labels.append(int(traj['time'][i].dt.day.values))
# plt.xticks(x_list, labels)
#
# print(x_list)
# print(labels)
# ax.legend()
# ax.set_ylabel('[hPa]')
# plt.savefig(f"{path_plots}/temp_height_traj_AC3.png", bbox_inches='tight', transparent=True)
# plt.show()
#
#
#
# # fig = plt.figure()
# # theta_im = plt.contourf(traj['time_ind'], ta['level'],  theta_height_traj, levels = 20, cmap = 'coolwarm')
# # plt.vlines(traj['time_ind'].sel(time = '2022-08-16T14', method = 'nearest'), ymin = level_min, ymax = 1000, colors = 'black', linestyles = 'dashed', label = 'Neutral core')
# # plt.vlines(traj['time_ind'].sel(time = '2022-08-19T14', method = 'nearest'), ymin = level_min, ymax = 1000, colors = 'black', linestyles = 'dashed', label = 'Cold core')
# # plt.gca().invert_yaxis()
# #
# # cbar = fig.colorbar(theta_im, orientation='vertical', fraction=0.046, pad=0.04)
# cbar.set_label(r'Theta[K]')
# # juste afficher les jours en xthicks
# x_list = []
# labels = []
# for i in range(len(traj['time'])) :
#     if int(traj['time'][i].dt.hour.values) == 0 :
#         x_list.append(int(traj['time_ind'][i].values))
#         labels.append(int(traj['time'][i].dt.day.values))
# plt.xticks(x_list, labels)
#
#
# ax.legend()
# ax.set_ylabel('[hPa]')
# plt.savefig(f"{path_plots}/theta_height_traj_AC3.png", bbox_inches='tight', transparent=True)
# plt.show()

