# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 16:54:13 2018

@author: bramv
"""
#%%
import numpy as np
import matplotlib.pyplot as plt

import read_cabauw_data as r
import calculate_geostrophic_wind as gw



#%%
months = list(range(1, 9))
#months = [8]
data = r.read_and_process_cabauw_data(months = months)
gw_data = gw.calculate_geostrophic_wind(months = months)



#%%
#Calculate the angle between the wind at 10m and the geostrophic wind
alpha = 180./np.pi * np.arccos(np.sum(data.V[:,:,-2] * gw_data.V_g, axis = -1) / (data.speed[:,:,-2] * gw_data.V_g_speed))
not_nan = np.isnan(alpha) == False
alpha = alpha[not_nan]
Vg_speed = gw_data.V_g_speed[not_nan]
dtheta = (data.theta[:,:,0] - data.theta[:,:,-2])[not_nan] #Difference in theta between 10 and 200 m



max_deltaV = alpha.max()
bin_size_deltaV = 1
n_bins_deltaV = int(max_deltaV / bin_size_deltaV)



max_Vg_speed = Vg_speed.max()
bin_size_Vgspeed = 2.5 #In m/s
n_bins_Vgspeed = int(np.ceil(max_Vg_speed / bin_size_Vgspeed))
alpha_mean_Vgspeed_binned = np.zeros((n_bins_Vgspeed, 2)) #First element of second axis gives mean geostrophic wind speed for the bin, second gives the mean alpha
alpha_sampledensity_Vgspeed = np.zeros(len(alpha))
for i in range(n_bins_Vgspeed):
    alpha_mean_Vgspeed_binned[i, 0] = (i + 0.5) * bin_size_Vgspeed
    alphas_in_bin = (Vg_speed >= i * bin_size_Vgspeed) & (Vg_speed < (i+1) * bin_size_Vgspeed)
    alpha_mean_Vgspeed_binned[i, 1] = np.mean(alpha[alphas_in_bin]) if np.count_nonzero(alphas_in_bin) >= 10 else np.nan
    for j in range(n_bins_deltaV):
        alphas_in_bin_j = alphas_in_bin & (alpha >= j * bin_size_deltaV) & (alpha < (j+1) * bin_size_deltaV)
        alpha_sampledensity_Vgspeed[alphas_in_bin_j] = np.count_nonzero(alphas_in_bin_j)
    


min_dtheta = dtheta.min(); max_dtheta = dtheta.max()
bin_size_dtheta = 1 #K
n_bins_dtheta = int(np.ceil((max_dtheta - min_dtheta) / bin_size_dtheta))
bins_min_dtheta = int(np.floor(min_dtheta/bin_size_dtheta)) * bin_size_dtheta

alpha_mean_dtheta_binned = np.zeros((n_bins_dtheta, 2))
alpha_sampledensity_dtheta = np.zeros(len(alpha))
for i in range(n_bins_dtheta):
    alpha_mean_dtheta_binned[i, 0] = bins_min_dtheta + (i + 0.5) * bin_size_dtheta
    alphas_in_bin = (dtheta >= bins_min_dtheta + i * bin_size_dtheta) & (dtheta < bins_min_dtheta + (i+1) * bin_size_dtheta)
    alpha_mean_dtheta_binned[i, 1] = np.mean(alpha[alphas_in_bin]) if np.count_nonzero(alphas_in_bin) >= 10 else np.nan
    for j in range(n_bins_deltaV):
        alphas_in_bin_j = alphas_in_bin & (alpha >= j * bin_size_deltaV) & (alpha < (j+1) * bin_size_deltaV)
        alpha_sampledensity_dtheta[alphas_in_bin_j] = np.count_nonzero(alphas_in_bin_j) 



#Bin alpha as a function of both Vg_speed and dtheta.
#The 4 elements in the last dimension of alpha_mean_Vgspeed_dtheta_binned are the mean Vgspeed,
#the mean dtheta, the number of alpha values and the mean alpha value in the bin.
alpha_mean_Vgspeed_dtheta_binned = np.zeros((n_bins_Vgspeed, n_bins_dtheta, 4))
for i in range(n_bins_Vgspeed):
    for j in range(n_bins_dtheta):
        alpha_mean_Vgspeed_dtheta_binned[i, j, 0] = (i + 0.5) * bin_size_Vgspeed
        alpha_mean_Vgspeed_dtheta_binned[i, j, 1] = bins_min_dtheta + (j + 0.5) * bin_size_dtheta
        alphas_in_bin = (Vg_speed >= i * bin_size_Vgspeed) & (Vg_speed < (i+1) * bin_size_Vgspeed) & (dtheta >= bins_min_dtheta + j * bin_size_dtheta) & (dtheta < bins_min_dtheta + (j+1) * bin_size_dtheta)
        alpha_mean_Vgspeed_dtheta_binned[i, j, 2] = np.count_nonzero(alphas_in_bin)
        #Calculate the mean only when there are at least 10 values to average over
        alpha_mean_Vgspeed_dtheta_binned[i, j, 3] = np.mean(alpha[alphas_in_bin]) if alpha_mean_Vgspeed_dtheta_binned[i, j, 2] >= 10 else np.nan



fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax[0].scatter(Vg_speed, alpha, c = alpha_sampledensity_Vgspeed)
ax[0].plot(alpha_mean_Vgspeed_binned[:,0], alpha_mean_Vgspeed_binned[:,1], 'r-')
ax[1].scatter(dtheta, alpha, c = alpha_sampledensity_dtheta)
ax[1].plot(alpha_mean_dtheta_binned[:,0], alpha_mean_dtheta_binned[:,1], 'r-')
ax[0].set_xlabel('$||\mathbf{V_g}||$ (m/s)'); ax[0].set_ylabel('$|\\alpha|$ $(\degree)$')
ax[1].set_xlabel('$\\theta$ (10 m) - $\\theta$ (200 m) (K)'); ax[1].set_ylabel('$|\\alpha|$ $(\degree)$')
plt.show()

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
im = ax[0].imshow(alpha_mean_Vgspeed_dtheta_binned[:,:,2].T, extent = [0, n_bins_Vgspeed * bin_size_Vgspeed, bins_min_dtheta + n_bins_dtheta * bin_size_dtheta, bins_min_dtheta], aspect = 'auto', cmap = 'jet')
plt.colorbar(im, ax = ax[0], orientation = 'horizontal', label = '# of samples per bin')
im = ax[1].imshow(alpha_mean_Vgspeed_dtheta_binned[:,:,3].T, extent = [0, n_bins_Vgspeed * bin_size_Vgspeed, bins_min_dtheta + n_bins_dtheta * bin_size_dtheta, bins_min_dtheta], aspect = 'auto', cmap = 'jet')
plt.colorbar(im, ax = ax[1], orientation = 'horizontal', label = '$|\\alpha|$ $(\degree)$')
ax[0].set_xlabel('$||\mathbf{V_g}||$ (m/s)'); ax[0].set_ylabel('$\\theta$ (10 m) - $\\theta$ (200 m) (K)', labelpad = -5)
ax[1].set_xlabel('$||\mathbf{V_g}||$ (m/s)'); ax[1].set_ylabel('$\\theta$ (10 m) - $\\theta$ (200 m) (K)', labelpad = -5)
plt.show()