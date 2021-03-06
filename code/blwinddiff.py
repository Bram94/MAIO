# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 15:36:29 2018

@author: bramv
"""
import numpy as np
import matplotlib.pyplot as plt

import read_cabauw_data as r
import calculate_geostrophic_wind as gw



months = list(range(1, 9))
data = r.read_and_process_cabauw_data(months = months)
gw_data = gw.calculate_geostrophic_wind(months = months)

delta_V = np.linalg.norm(data.V[:,:,0] - data.V[:,:,-2], axis = 2).flatten()
Vg_speed = gw_data.V_g_speed.flatten()
dtheta = (data.theta[:,:,0] - data.theta[:,:,-2]).flatten() #Difference in theta between 10 and 200 m



max_deltaV = delta_V.max()
bin_size_deltaV = 1
n_bins_deltaV = int(max_deltaV / bin_size_deltaV)



max_Vg_speed = Vg_speed.max()
bin_size_Vgspeed = 2.5 #In m/s
n_bins_Vgspeed = int(np.ceil(max_Vg_speed / bin_size_Vgspeed))
delta_V_mean_Vgspeed_binned = np.zeros((n_bins_Vgspeed, 2)) #First element of second axis gives mean geostrophic wind speed for the bin, second gives the mean delta_V
delta_V_sampledensity_Vgspeed = np.zeros(len(delta_V))
for i in range(n_bins_Vgspeed):
    delta_V_mean_Vgspeed_binned[i, 0] = (i + 0.5) * bin_size_Vgspeed
    delta_Vs_in_bin = (Vg_speed >= i * bin_size_Vgspeed) & (Vg_speed < (i+1) * bin_size_Vgspeed)
    delta_V_mean_Vgspeed_binned[i, 1] = np.mean(delta_V[delta_Vs_in_bin]) if np.count_nonzero(delta_Vs_in_bin) >= 10 else np.nan
    for j in range(n_bins_deltaV):
        delta_Vs_in_bin_j = delta_Vs_in_bin & (delta_V >= j * bin_size_deltaV) & (delta_V < (j+1) * bin_size_deltaV)
        delta_V_sampledensity_Vgspeed[delta_Vs_in_bin_j] = np.count_nonzero(delta_Vs_in_bin_j)
    


min_dtheta = dtheta.min(); max_dtheta = dtheta.max()
bin_size_dtheta = 1 #K
n_bins_dtheta = int(np.ceil((max_dtheta - min_dtheta) / bin_size_dtheta))
bins_min_dtheta = int(np.floor(min_dtheta/bin_size_dtheta)) * bin_size_dtheta

delta_V_mean_dtheta_binned = np.zeros((n_bins_dtheta, 2))
delta_V_sampledensity_dtheta = np.zeros(len(delta_V))
for i in range(n_bins_dtheta):
    delta_V_mean_dtheta_binned[i, 0] = bins_min_dtheta + (i + 0.5) * bin_size_dtheta
    delta_Vs_in_bin = (dtheta >= bins_min_dtheta + i * bin_size_dtheta) & (dtheta < bins_min_dtheta + (i+1) * bin_size_dtheta)
    delta_V_mean_dtheta_binned[i, 1] = np.mean(delta_V[delta_Vs_in_bin]) if np.count_nonzero(delta_Vs_in_bin) >= 10 else np.nan
    for j in range(n_bins_deltaV):
        delta_Vs_in_bin_j = delta_Vs_in_bin & (delta_V >= j * bin_size_deltaV) & (delta_V < (j+1) * bin_size_deltaV)
        delta_V_sampledensity_dtheta[delta_Vs_in_bin_j] = np.count_nonzero(delta_Vs_in_bin_j) 



#Bin delta_V as a function of both Vg_speed and dtheta.
#The 4 elements in the last dimension of delta_V_mean_Vgspeed_dtheta_binned are the mean Vgspeed,
#the mean dtheta, the number of delta_V values and the mean delta_V value in the bin.
delta_V_mean_Vgspeed_dtheta_binned = np.zeros((n_bins_Vgspeed, n_bins_dtheta, 4))
for i in range(n_bins_Vgspeed):
    for j in range(n_bins_dtheta):
        delta_V_mean_Vgspeed_dtheta_binned[i, j, 0] = (i + 0.5) * bin_size_Vgspeed
        delta_V_mean_Vgspeed_dtheta_binned[i, j, 1] = bins_min_dtheta + (j + 0.5) * bin_size_dtheta
        delta_Vs_in_bin = (Vg_speed >= i * bin_size_Vgspeed) & (Vg_speed < (i+1) * bin_size_Vgspeed) & (dtheta >= bins_min_dtheta + j * bin_size_dtheta) & (dtheta < bins_min_dtheta + (j+1) * bin_size_dtheta)
        delta_V_mean_Vgspeed_dtheta_binned[i, j, 2] = np.count_nonzero(delta_Vs_in_bin)
        #Calculate the mean only when there are at least 10 values to average over
        delta_V_mean_Vgspeed_dtheta_binned[i, j, 3] = np.mean(delta_V[delta_Vs_in_bin]) if delta_V_mean_Vgspeed_dtheta_binned[i, j, 2] >= 10 else np.nan



fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax[0].scatter(Vg_speed, delta_V, c = delta_V_sampledensity_Vgspeed)
ax[0].plot(delta_V_mean_Vgspeed_binned[:,0], delta_V_mean_Vgspeed_binned[:,1], 'r-')
ax[1].scatter(dtheta, delta_V, c = delta_V_sampledensity_dtheta)
ax[1].plot(delta_V_mean_dtheta_binned[:,0], delta_V_mean_dtheta_binned[:,1], 'r-')
ax[0].set_xlabel('$||\mathbf{V_g}||$ (m/s)'); ax[0].set_ylabel('$||\mathbf{V}$ (200 m) - $\mathbf{V}$ (10 m)$||$ (m/s)')
ax[1].set_xlabel('$\\theta$ (10 m) - $\\theta$ (200 m) (K)'); ax[1].set_ylabel('$||\mathbf{V}$ (200 m) - $\mathbf{V}$ (10 m)$||$ (m/s)')
plt.show()

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
im = ax[0].imshow(delta_V_mean_Vgspeed_dtheta_binned[:,:,2].T, extent = [0, n_bins_Vgspeed * bin_size_Vgspeed, bins_min_dtheta + n_bins_dtheta * bin_size_dtheta, bins_min_dtheta], aspect = 'auto', cmap = 'jet')
plt.colorbar(im, ax = ax[0], orientation = 'horizontal', label = '# of samples per bin')
im = ax[1].imshow(delta_V_mean_Vgspeed_dtheta_binned[:,:,3].T, extent = [0, n_bins_Vgspeed * bin_size_Vgspeed, bins_min_dtheta + n_bins_dtheta * bin_size_dtheta, bins_min_dtheta], aspect = 'auto', cmap = 'jet')
plt.colorbar(im, ax = ax[1], orientation = 'horizontal', label = '$||\mathbf{V}$ (200 m) - $\mathbf{V}$ (10 m)$||$ (m/s)')
ax[0].set_xlabel('$||\mathbf{V_g}||$ (m/s)'); ax[0].set_ylabel('$\\theta$ (10 m) - $\\theta$ (200 m) (K)', labelpad = -5)
ax[1].set_xlabel('$||\mathbf{V_g}||$ (m/s)'); ax[1].set_ylabel('$\\theta$ (10 m) - $\\theta$ (200 m) (K)', labelpad = -5)
plt.show()