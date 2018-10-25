# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 11:39:32 2018

@author: angel
"""
#%%
import calendar
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import settings as s
import functions as ft
import read_cabauw_data as r
import calculate_geostrophic_wind as gw
import calculate_temperature_gradient as tg


mpl.rcParams['axes.titlesize'] = 22
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['axes.labelsize'] = 22
mpl.rcParams['axes.labelweight'] = 'bold'
for j in ('xtick','ytick'):
    mpl.rcParams[j+'.labelsize'] = 18
mpl.rcParams['legend.fontsize'] = 18



#%%
months = [12] + list(range(1,13)) * 6 + list(range(1, 9))
years = [2011] + [2012] * 12 + [2013] * 12 + [2014] * 12 + [2015] * 12 + [2016] * 12 + [2017] * 12 + [2018] * 8
#months = [1]
data = r.read_and_process_cabauw_data(years, months)
gw_data = gw.calculate_geostrophic_wind(years, months)
tg_data = tg.calculate_temperature_gradient(years, months)


"""Plots are now created for the period from 12 to 12 UTC, instead of 0 to 0 UTC.
For a given date, the time range is from 12 UTC at the previous date to 12 UTC at the given date.
In order to plot the data for this time range, all datasets are below shifted backward in time by
12 hours.
"""
n_days_firstmonth = calendar.monthrange(int(years[0]), int(months[0]))[1]
n_days_remainingmonths = sum([calendar.monthrange(years[i], months[i])[1] for i in range(1, len(months))])
shift = 72
for j in data.__dict__:
    if len(eval('data.'+j).shape) >=2:
        exec('data.'+j+'= np.reshape(data.'+j+', (data.'+j+'.shape[0] * data.'+j+'.shape[1],) + (data.'+j+'.shape[2:] if len(data.'+j+'.shape) > 2 else ()))')
        exec('data.'+j+'=data.'+j+'[n_days_firstmonth * 144 - shift: - shift]')
        exec('data.'+j+'=np.reshape(data.'+j+', (n_days_remainingmonths, 144) + (data.'+j+'.shape[1:] if len(data.'+j+'.shape) > 1 else ()))')

for j in gw_data.__dict__:
    if len(eval('gw_data.'+j).shape) >=2:
        exec('gw_data.'+j+'= np.reshape(gw_data.'+j+', (gw_data.'+j+'.shape[0] * gw_data.'+j+'.shape[1],) + (gw_data.'+j+'.shape[2:] if len(gw_data.'+j+'.shape) > 2 else ()))')
        exec('gw_data.'+j+'=gw_data.'+j+'[n_days_firstmonth * 144 - shift: - shift]')
        exec('gw_data.'+j+'=np.reshape(gw_data.'+j+', (n_days_remainingmonths, 144) + (gw_data.'+j+'.shape[1:] if len(gw_data.'+j+'.shape) > 1 else ()))')

for j in tg_data.__dict__:
    if len(eval('tg_data.'+j).shape) >=2:
        exec('tg_data.'+j+'= np.reshape(tg_data.'+j+', (tg_data.'+j+'.shape[0] * tg_data.'+j+'.shape[1],) + (tg_data.'+j+'.shape[2:] if len(tg_data.'+j+'.shape) > 2 else ()))')
        exec('tg_data.'+j+'=tg_data.'+j+'[n_days_firstmonth * 144 - shift: - shift]')
        exec('tg_data.'+j+'=np.reshape(tg_data.'+j+', (n_days_remainingmonths, 144) + (tg_data.'+j+'.shape[1:] if len(tg_data.'+j+'.shape) > 1 else ()))')



hour_filter = (data.hours >= 18) | (data.hours <= 6)

for j in data.__dict__.copy():
    if len(eval('data.'+j).shape) >=2:
        exec('data.'+j+'_18to6utc = np.reshape(data.'+j+'[hour_filter], (n_days_remainingmonths, 73) + (data.'+j+'.shape[2:] if len(data.'+j+'.shape) > 2 else ()))')

for j in gw_data.__dict__.copy():
    if len(eval('gw_data.'+j).shape) >=2:
        exec('gw_data.'+j+'_18to6utc = np.reshape(gw_data.'+j+'[hour_filter], (n_days_remainingmonths, 73) + (gw_data.'+j+'.shape[2:] if len(gw_data.'+j+'.shape) > 2 else ()))')

for j in tg_data.__dict__.copy():
    if len(eval('tg_data.'+j).shape) >=2:
        exec('tg_data.'+j+'_18to6utc = np.reshape(tg_data.'+j+'[hour_filter], (n_days_remainingmonths, 73) + (tg_data.'+j+'.shape[2:] if len(tg_data.'+j+'.shape) > 2 else ()))')

        
        
#%%
        
#Calculate the geostrophic wind and rotate the vectors to the east
 #rotate geostrophic wind vectors to the east
g = 9.81
R = 287.
f = 2 * 7.29e-5 * np.sin(gw_data.coords_cabauw[1] * np.pi/180.)

def f_dpdz_dTdz(y): #y = np.array([p, T])
    p, T = y
    dpdz = -g/R * p/T
    return np.array([dpdz, dTdz]) #dTdz is defined in the loop below
    
p_0 = data.p0_18to6utc * 100. #Pressure at 1.5 m
p_200m = p_0.copy()
for i in range(1, len(data.z)):
    z_old = data.z[-i]; z_new = data.z[-(i+1)]
    dz = z_new - z_old
    dTdz = (data.T_18to6utc[:,:,-(i+1)] - data.T_18to6utc[:,:,-i]) / dz
    p_200m += ft.RK4(f_dpdz_dTdz, np.array([p_200m, data.T_18to6utc[:,:,-i]]), dz)[0]
V_thermal_18to6utc = R / f * np.array([[[-1, 1]]]) * tg_data.T_gradient_xy_18to6utc[:,:,::-1] * np.log(p_200m / p_0)[:,:,np.newaxis]


# Find the geostrophic wind, rotate the vectors to the east and normalize the wind vectors with the geostrophic wind
Vg = gw_data.V_g_18to6utc[:,:,np.newaxis,:] + (data.z[np.newaxis, np.newaxis, :-1, np.newaxis] - 1.5) / 198.5 * V_thermal_18to6utc[:,:,np.newaxis,:]
Vg_speed = np.linalg.norm(Vg, axis = 3)
V = data.V_18to6utc[:,:,:-1]
 


dates = gw_data.dates[:, -1]
months = np.array([int(j[4:6]) for j in dates.astype('str')])

net_longwave = data.longwave_upward - data.longwave_downward

dtheta = (data.theta[:,:,0] - data.theta[:,:,-2])#.flatten() #Difference in theta between 10 and 200 m

Vg_daymean_speed = np.linalg.norm(np.mean(Vg, axis = (1,2)), axis = -1)

Vg_hours = Vg[:, ::6] 
Vg_diffs = np.zeros((Vg_hours.shape[0], Vg_hours.shape[1]**2, Vg_hours.shape[2]))
for j in range(Vg_hours.shape[1]):
    Vg_diffs[:, Vg_hours.shape[1]*j : Vg_hours.shape[1] * (j+1)] = np.linalg.norm(Vg_hours[:,j][:,np.newaxis,:,:] - Vg_hours, axis = -1)
Vg_diffs_daymaxes = np.max(Vg_diffs, axis = 1)


month_criterion = (months >= 5) & (months <= 7) #Take the months May, June and July, to have a relatively constant daylight period
longwave_criterion = (np.min(net_longwave, axis = 1) > 20) #Net upward longwave radiation > 20 W/m^2 during the whole 24 hours
dtheta_criterion = (dtheta[:,0] < 0) & (dtheta[:, 72] > 3) #dtheta(12Z) < 0 K, dtheta(00Z) > 3K
Vgspeed_criterion = (Vg_daymean_speed >= 5) & (Vg_daymean_speed <= 15) #Norm of vector-averaged geostrophic wind from 18-6 UTC should be between 5 and 15 m/s
Vgdiff_criterion = np.max(Vg_diffs_daymaxes, axis = 1) <= 6 #Maximum norm of the difference in geostrophic wind between any 2 times should be less than 6 m/s, 
#for all 6 heights

combi_criterion = month_criterion & longwave_criterion & dtheta_criterion & Vgspeed_criterion & Vgdiff_criterion
#combi_criterion = np.ones(V.shape[0], dtype = 'bool')
print('n_dates = ', np.count_nonzero(combi_criterion))



V_filtered = V[combi_criterion]
Vg_filtered = Vg[combi_criterion]
Vg_speed_filtered = Vg_speed[combi_criterion]
V_daymean_filtered = np.zeros(V_filtered.shape)
V_daymean_filtered[:,:] = np.mean(V_filtered, axis = 1)[:,np.newaxis]
Vg_daymean_filtered = np.zeros(V_filtered.shape)
Vg_daymean_speed_filtered = np.zeros(Vg_speed_filtered.shape)
for j in range(V_filtered.shape[0]):
    Vg_daymean_filtered[j,:] = np.mean(Vg_filtered[j], axis = 0)
    Vg_daymean_speed_filtered[j,:] = np.mean(Vg_speed_filtered[j], axis = 0)



"""Normalize V by rotating the wind profile in such a way that the geostrophic wind vector points to the east, and
scaling the profile by the geostrophic wind speed.
"""
angle_rotate = np.arctan2(Vg_daymean_filtered[:,:,:,1], Vg_daymean_filtered[:,:,:,0])
rot_matrix = np.zeros(V_filtered.shape[:3] + (2, 2))
rot_matrix[:,:,:,0,0] = np.cos(angle_rotate); rot_matrix[:,:,:,0,1] = np.sin(angle_rotate)
rot_matrix[:,:,:,1,0] = - np.sin(angle_rotate); rot_matrix[:,:,:,1,1] = np.cos(angle_rotate)
normalized_V = np.matmul(V_filtered[:,:,:,np.newaxis,:], np.transpose(rot_matrix, axes = [0,1,2,4,3]))[:,:,:,0] / Vg_daymean_speed_filtered[:,:,:,np.newaxis]


plt.figure(figsize = (15, 10))
colors = ['green', 'yellow', 'red', 'pink', 'purple', 'blue']
height_indices = list(range(len(data.z) - 1))
for i in range(len(height_indices)): #Plot hodographs at 10, 80 and 200 m
    j = height_indices[i]
    mean_profile = np.mean(normalized_V[:,:,j], axis = 0)
    plt.plot(mean_profile[:, 0], mean_profile[:, 1], colors[i], linewidth = 3)
plt.grid()
plt.xlabel('u / G'); plt.ylabel('v / G')
plt.title('Normalized wind profiles from 18 to 06 UTC based on '+str(np.count_nonzero(combi_criterion))+' cases')
plt.legend([str(int(data.z[j]))+' m' for j in height_indices])
plt.savefig(s.imgs_path+'inertial_oscillation.jpg', dpi = 120, bbox_inches = 'tight')
plt.show()

plt.figure(figsize = (15, 10))
for j in range(normalized_V.shape[0]):
    plt.plot(normalized_V[j,:,0,0], normalized_V[j, :, 0, 1])
plt.legend([str(int(j)) for j in dates[combi_criterion]])
plt.show()