# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 11:48:48 2018

@author: angel
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

g = 9.81
Cp = 1005

f = xr.open_dataset('cesar_tower_meteo_lc1_t10_v1.0_201808.nc', decode_times = False)
print(f.variables)

time = np.array(f.variables['time']) #Is given in hours, with data every 10 minutes
z = np.array(f.variables['z'])
speed = np.array(f.variables['F'])
direction = np.array(f.variables['D'])
T = np.array(f.variables['TA'])

n_time = len(time); n_z = len(z)


#Reshape the data by adding an extra axis that represents different days
n_times_day = 6*24 #Data every 10 minutes
n_days = int(n_time / n_times_day)
hours = np.reshape(np.mod(time, 24), (n_days, n_times_day))
speed = np.reshape(speed, (n_days, n_times_day, n_z))
direction = np.reshape(direction, speed.shape)
u = speed * np.sin(direction * np.pi/180.)
v = speed * np.cos(direction * np.pi/180.)
T = np.reshape(T, speed.shape)
theta = T + g/Cp*z[np.newaxis, np.newaxis, :]

direction10m = np.zeros(n_days)
direction200m = np.zeros(n_days)

direction1aug10m = direction[0,0,5]
direction1aug200m = direction[0,0,0]
dalpha = direction1aug200m - direction1aug10m
print(dalpha)

direction2aug10m = direction[1,0,5]
direction2aug200m = direction[1,0,0]
dalpha = direction2aug200m - direction2aug10m
print(dalpha)
    
print(direction[:,:,0])
print(direction[:,:,-2])
dalpha = direction[:,:,0] - direction[:,:,-2]


theta200m = theta[:,:,0]
theta10m = theta[:,:,-2]
dtheta = theta200m - theta10m

plt.figure()
plt.plot(dalpha)
plt.show()

plt.figure()
plt.plot(dtheta[0])
plt.show()
