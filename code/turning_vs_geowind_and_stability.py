# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 16:54:13 2018

@author: bramv
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import settings as s
import read_cabauw_data as r
import calculate_geostrophic_wind as gw


mpl.rcParams['axes.titlesize'] = 22
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['axes.labelsize'] = 22
mpl.rcParams['axes.labelweight'] = 'bold'
for j in ('xtick','ytick'):
    mpl.rcParams[j+'.labelsize'] = 18
mpl.rcParams['legend.fontsize'] = 18



#%%
months = list(range(1,13)) * 9 + list(range(1, 9))
years = [2009] * 12 + [2010] * 12 + [2011] * 12 + [2012] * 12 + [2013] * 12 + [2014] * 12 + [2015] * 12 + [2016] * 12 + [2017] * 12 + [2018] * 8
#months = [1]
data = r.read_and_process_cabauw_data(years, months)
gw_data = gw.calculate_geostrophic_wind(years, months)



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



n_samples_min = 50

max_Vg_speed = Vg_speed.max()
bin_size_Vgspeed = 2.5 #In m/s
n_bins_Vgspeed = int(np.ceil(max_Vg_speed / bin_size_Vgspeed))
alpha_mean_Vgspeed_binned = np.zeros((n_bins_Vgspeed, 2)) #First element of second axis gives mean geostrophic wind speed for the bin, second gives the mean alpha
alpha_sampledensity_Vgspeed = np.zeros(len(alpha))
for i in range(n_bins_Vgspeed):
    alpha_mean_Vgspeed_binned[i, 0] = (i + 0.5) * bin_size_Vgspeed
    alphas_in_bin = (Vg_speed >= i * bin_size_Vgspeed) & (Vg_speed < (i+1) * bin_size_Vgspeed)
    alpha_mean_Vgspeed_binned[i, 1] = np.mean(alpha[alphas_in_bin]) if np.count_nonzero(alphas_in_bin) >= n_samples_min else np.nan
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
    alpha_mean_dtheta_binned[i, 1] = np.mean(alpha[alphas_in_bin]) if np.count_nonzero(alphas_in_bin) >= n_samples_min else np.nan
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
        #Calculate the mean only when there are at least n_samples_min values to average over
        alpha_mean_Vgspeed_dtheta_binned[i, j, 3] = np.mean(alpha[alphas_in_bin]) if alpha_mean_Vgspeed_dtheta_binned[i, j, 2] >= n_samples_min else np.nan



#%%
fig, ax = plt.subplots(2, 2, figsize = (20, 20))
ax[0, 0].scatter(Vg_speed, alpha, c = alpha_sampledensity_Vgspeed)
ax[0, 0].plot(alpha_mean_Vgspeed_binned[:,0], alpha_mean_Vgspeed_binned[:,1], 'r-', linewidth = 5)
ax[0, 1].scatter(dtheta, alpha, c = alpha_sampledensity_dtheta)
ax[0, 1].plot(alpha_mean_dtheta_binned[:,0], alpha_mean_dtheta_binned[:,1], 'r-', linewidth = 5)
ax[0, 0].set_xlabel('$\mathbf{||V_g||}$ (m/s)'); ax[0, 0].set_ylabel('$\mathbf{|\\alpha|}$ $\mathbf{(\degree)}$')
ax[0, 1].set_xlabel('$\mathbf{\\theta}$ (10 m) - $\mathbf{\\theta}$ (200 m) (K)'); ax[0, 1].set_ylabel('$\mathbf{|\\alpha|}$ $\mathbf{(\degree)}$')

im = ax[1, 0].imshow(alpha_mean_Vgspeed_dtheta_binned[:,:,2].T / np.sum(alpha_mean_Vgspeed_dtheta_binned[:,:,2]), extent = [0, n_bins_Vgspeed * bin_size_Vgspeed, bins_min_dtheta + n_bins_dtheta * bin_size_dtheta, bins_min_dtheta], aspect = 'auto', cmap = 'jet')
plt.colorbar(im, ax = ax[1, 0], orientation = 'horizontal', label = 'Fraction of samples per bin')
im = ax[1, 1].imshow(alpha_mean_Vgspeed_dtheta_binned[:,:,3].T, extent = [0, n_bins_Vgspeed * bin_size_Vgspeed, bins_min_dtheta + n_bins_dtheta * bin_size_dtheta, bins_min_dtheta], aspect = 'auto', cmap = 'jet')
plt.colorbar(im, ax = ax[1, 1], orientation = 'horizontal', label = '$\mathbf{|\\alpha|}$ $\mathbf{(\degree)}$')
ax[1, 0].set_xlabel('$\mathbf{||V_g||}$ (m/s)'); ax[1, 0].set_ylabel('$\mathbf{\\theta}$ (10 m) - $\mathbf{\\theta}$ (200 m) (K)', labelpad = -5)
ax[1, 1].set_xlabel('$\mathbf{||V_g||}$ (m/s)'); ax[1, 1].set_ylabel('$\mathbf{\\theta}$ (10 m) - $\mathbf{\\theta}$ (200 m) (K)', labelpad = -5)
plt.subplots_adjust(hspace = 0.12, wspace = 0.15)
plt.savefig(s.imgs_path+'turning_angle.jpg', dpi = 120, bbox_inches = 'tight')
plt.show()