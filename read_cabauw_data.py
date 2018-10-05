# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 14:33:36 2018

@author: bramv
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

if False:
    f = xr.open_dataset('cesar_tower_meteo_lc1_t10_v1.0_201808.nc', decode_times = False)
    time = np.array(f.variables['time']) #Is given in hours, with data every 10 minutes
    z = np.array(f.variables['z'])
    speed = np.array(f.variables['F'])
    direction = np.array(f.variables['D'])
    
    n_time = len(time); n_z = len(z)
    
    #Reshape the data by adding an extra axis that represents different days
    n_times_day = 6*24 #Data every 10 minutes
    n_days = int(n_time / n_times_day)
    hours = np.reshape(np.mod(time, 24), (n_days, n_times_day))
    speed = np.reshape(speed, (n_days, n_times_day, n_z))
    direction = np.reshape(direction, speed.shape)
    u = speed * np.sin(direction * np.pi/180.)
    v = speed * np.cos(direction * np.pi/180.)

fig, ax = plt.subplots(6,5, figsize = (20,20))
time_index = -1
for j in range(len(ax.flat)):
    u_j, v_j = u[j, time_index, :-1], v[j, time_index, :-1] #Exclude the last element as it is np.nan
    ax.flat[j].plot(u_j, v_j, 'bo', u_j, v_j, 'b-')
    u_min = u[j,time_index,:-1].min(); u_max = u_j.max()
    v_min = v[j,time_index,:-1].min(); v_max = v_j.max()
    ax.flat[j].set_xlim([min(-1, u_min - 1), max(1, u_max + 1)])
    ax.flat[j].set_ylim([min(-1, v_min - 1), max(1, v_max + 1)])
    for k in range(len(u_j)):
        ax.flat[j].text(u_j[k], v_j[k], str(int(z[k])))
plt.show()

