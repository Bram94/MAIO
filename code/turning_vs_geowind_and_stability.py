# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 16:54:13 2018

@author: bramv
"""
import numpy as np
import matplotlib.pyplot as plt

import read_cabauw_data as r
import calculate_geostrophic_wind as gw
import settings as s



data = r.read_and_process_cabauw_data()
gw_data = gw.calculate_geostrophic_wind()


#Calculate the angle between the wind at 10m and the geostrophic wind
alpha = 180./np.pi * np.arccos(np.sum(data.V[:,:,-2] * gw_data.V_g, axis = -1) / (data.speed[:,:,-2] * gw_data.V_g_speed))
not_nan = np.isnan(alpha) == False
alpha = alpha[not_nan]
Vg_speed = gw_data.V_g_speed[not_nan]

max_Vg_speed = Vg_speed.max()
bin_size = 2.5 #In m/s
n_bins = int(np.ceil(max_Vg_speed / bin_size))
alpha_mean_Vgspeed_binned = np.zeros((n_bins, 2)) #First element of second axis gives mean geostrophic wind speed for the bin, second gives the mean alpha
for i in range(n_bins):
    alpha_mean_Vgspeed_binned[i, 0] = (i + 0.5) * bin_size
    alpha_mean_Vgspeed_binned[i, 1] = np.mean(alpha[(Vg_speed >= i * bin_size) & (Vg_speed < (i+1) * bin_size)])



dtheta = data.theta[:,:,0] - data.theta[:,:,-1] #Difference in theta between 2 and 200 m
dtheta = dtheta[not_nan]
min_dtheta = dtheta.min(); max_dtheta = dtheta.max()
bin_size = 1 #K
n_bins = int(np.ceil((max_dtheta - min_dtheta) / bin_size))

alpha_mean_dtheta_binned = np.zeros((n_bins, 2))
for i in range(n_bins):
    alpha_mean_dtheta_binned[i, 0] = min_dtheta + (i + 0.5) * bin_size
    alpha_mean_dtheta_binned[i, 1] = np.mean(alpha[(dtheta >= min_dtheta + i * bin_size) & (dtheta < min_dtheta + (i+1) * bin_size)])


fig, ax = plt.subplots(1, 2, figsize = (14,6))
ax[0].plot(Vg_speed, alpha, 'bo', alpha_mean_Vgspeed_binned[:,0], alpha_mean_Vgspeed_binned[:,1], 'r-')
ax[1].plot(dtheta, alpha, 'bo', alpha_mean_dtheta_binned[:,0], alpha_mean_dtheta_binned[:,1], 'r-')
plt.show()