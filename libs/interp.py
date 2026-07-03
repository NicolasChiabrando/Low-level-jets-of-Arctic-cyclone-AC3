import numpy as np
from scipy.interpolate import RegularGridInterpolator
import xarray as xr


def interp_traj(n_seed, data, lat, lon, level, time, traj, hPa = False, varlev = 'P') :


    #lat_seed = traj['lat'].sel(n_seeds = n_seed).to_numpy()
    #lon_seed = traj['lon'].sel(n_seeds = n_seed).to_numpy()
    #lev_seed = traj[varlev].sel(n_seeds = n_seed).to_numpy() #* 100
    #time_seed = traj['time'].sel(n_seeds = n_seed).to_numpy()

    lat_seed = traj['lat'][n_seed].to_numpy()
    lon_seed = traj['lon'][n_seed].to_numpy()
    lev_seed = traj[varlev][n_seed].to_numpy()  # * 100
    time_seed = traj['time'][n_seed].to_numpy()

    if hPa :
        lev_seed = lev_seed * 100
    #print(P_seed)

    mask = np.isnan(lev_seed)

    T = np.size(time_seed)
    lev_seed = lev_seed[mask==False]
    lat_seed = lat_seed[mask==False]
    lon_seed=lon_seed[mask==False]
    time_seed=time_seed[mask==False]



    #T, L, La, Lo = np.meshgrid(time, level, lat, lon, indexing='ij')

    # Flatten the meshgrid and data
    #points = np.array([T.flatten(), L.flatten(), La.flatten(), Lo.flatten()]).T
    #values = data.flatten()


    interpolator = RegularGridInterpolator((time, level, lat, lon), data, fill_value=np.nan, bounds_error = False)
    #interpolator = RegularGridInterpolator(points, values, fill_value=np.nan)
    #print(interpolator)
    Position = np.column_stack((time_seed, lev_seed, lat_seed, lon_seed))
    data_interp=interpolator(Position)
    #data_interp = interpn((time, level, lat, lon), data, Position, fill_value=np.nan )



    data_result=np.full(T,np.nan)
    data_result[0:len(data_interp)]=data_interp
    #print(time_seed)
    return data_result