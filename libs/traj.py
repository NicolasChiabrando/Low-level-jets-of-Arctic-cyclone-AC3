import numpy as np
import xarray as xr

def is_in_region(traj_ini, cyc) :
    n_list = []

    if cyc == 'AC3A' :
        for i in traj_ini['n_seeds'].to_numpy() :
            if traj_ini.isel(time_ind=0).sel(n_seeds=i)['P'] >= 850 * 100:
                if traj_ini.isel(time_ind=0).sel(n_seeds=i)['lat'] < 77.5:
            # if traj_ini.isel(time_ind = 0).sel(n_seeds = i)['lon'] <= 21 or traj_ini.isel(time_ind = 0).sel(n_seeds = i)['lat'] <= 76 :
            #     # if traj_ini.isel(time_ind = 0).sel(n_seeds = i)['lon'] <= 23 :
            #if traj_ini.isel(time_ind = 0).sel(n_seeds = i)['lat'] <= 87.2 :
            # if traj_ini['pv'].sel(n_seeds = i).isel(time_ind = 0) > traj_ini['pv'].sel(n_seeds = i).isel(time_ind = -1) :
                    if traj_ini.isel(time_ind = 0).sel(n_seeds = i)['lat'] > 72 :
                        # if traj_ini.isel(time_ind = 0).sel(n_seeds = i)['lon'] < 20 :
                        #     if traj_ini.isel(time_ind=0).sel(n_seeds=i)['lat'] > 74.5 :
                        #             n_list.append(i)

                        if traj_ini.isel(time_ind = 0).sel(n_seeds = i)['lon'] > -3 :
                            #n_region1_list.append(i)
                            n_list.append(i)
                        elif traj_ini.isel(time_ind = 0).sel(n_seeds = i)['lat'] < 73 :
                         #   n_region2_list.append(i)
                            n_list.append(i)

                    elif traj_ini.isel(time_ind = 0).sel(n_seeds = i)['lon'] < 1 :
                        if traj_ini.isel(time_ind = 0).sel(n_seeds = i)['lat'] > 67 :
                         #   n_region2_list.append(i)
                            n_list.append(i)
            # if traj_ini.isel(time_ind = 0).sel(n_seeds = i)['lon'] <= 21 or traj_ini.isel(time_ind = 0).sel(n_seeds = i)['lat'] <= 76 :
            #     # if traj_ini.isel(time_ind = 0).sel(n_seeds = i)['lon'] <= 23 :
            #if traj_ini.isel(time_ind = 0).sel(n_seeds = i)['lat'] <= 87.2 :
            # if traj_ini['pv'].sel(n_seeds = i).isel(time_ind = 0) > traj_ini['pv'].sel(n_seeds = i).isel(time_ind = -1) :


    elif cyc == 'AC3B' :
        for i in traj_ini['n_seeds'].to_numpy() :
            if traj_ini.isel(time_ind = 0).sel(n_seeds = i)['P'] >= 850 * 100:
                if traj_ini.isel(time_ind = 0).sel(n_seeds = i)['lat'] <= 87.8 :
           #  #     # if traj_ini.isel(time_ind = 0).sel(n_seeds = i)['lon'] <= 23 :
           #      if traj_ini.isel(time_ind = 0).sel(n_seeds = i)['lat'] <= 87.2 :
           #     if traj_ini['pv'].sel(n_seeds = i).isel(time_ind = 0) > traj_ini['pv'].sel(n_seeds = i).isel(time_ind = -1) :
                    n_list.append(i)
    elif cyc == 'AC4' :
        for i in traj_ini['n_seeds'].to_numpy() :
            if  traj_ini.isel(time_ind = 0).sel(n_seeds = i)['lat'] <= 78.7 :
                if traj_ini.isel(time_ind = 0).sel(n_seeds = i)['lon'] <= 40 or traj_ini.isel(time_ind = 0).sel(n_seeds = i)['lat'] >= 76.5 :
            #     # if traj_ini.isel(time_ind = 0).sel(n_seeds = i)['lon'] <= 23 :
            # if traj_ini.isel(time_ind = 0).sel(n_seeds = i)['lat'] >= 76 :
            #         if traj_ini['pv'].sel(n_seeds = i).isel(time_ind = 0) > traj_ini['pv'].sel(n_seeds = i).isel(time_ind = -1) :
                      n_list.append(i)
    return n_list


def date_from_time_ind(time_ind, h0, d0):
    h = h0 - int(time_ind / 2 + time_ind % 2)
    d = d0

    while h < 0:
        h = 24 + h
        d = d0 - 1

    if d < 10:
        d_txt = f'0{d}'
    else:
        d_txt = str(d)
    if h < 10:
        h_txt = f'0{h}'
    else:
        h_txt = str(h)

    if time_ind % 2 == 0:
        time_cs = f'2022-08-{d}T{h_txt}:00'
    else:
        time_cs = f'2022-08-{d}T{h_txt}:30'

    return time_cs