# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 11:39:32 2018

@author: angel
"""
#%%
import calendar
import numpy as np
import matplotlib.pyplot as plt

import functions as ft
import read_cabauw_data as r
import calculate_geostrophic_wind as gw
import calculate_temperature_gradient as tg



#%%
months = [12] + list(range(1,9))
years = [2017] + [2018] * 8
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



"""
hour_filter = (data.hours >= 18) | (data.hours <= 6)

for j in data.__dict__:
    if len(eval('data.'+j).shape) >=2:
        exec('data.'+j+'_filtered = np.reshape(data.'+j+'[hour_filter], (n_days_remainingmonths, 73) + (data.'+j+'.shape[2:] if len(data.'+j+'.shape) > 2 else ()))')

for j in gw_data.__dict__:
    if len(eval('gw_data.'+j).shape) >=2:
        exec('gw_data.'+j+'_filtered = np.reshape(gw_data.'+j+'[hour_filter], (n_days_remainingmonths, 73) + (gw_data.'+j+'.shape[2:] if len(gw_data.'+j+'.shape) > 2 else ()))')

for j in tg_data.__dict__:
    if len(eval('tg_data.'+j).shape) >=2:
        exec('tg_data.'+j+'_filtered = np.reshape(tg_data.'+j+'[hour_filter], (n_days_remainingmonths, 73) + (tg_data.'+j+'.shape[2:] if len(tg_data.'+j+'.shape) > 2 else ()))')
"""        
        
        
#%%

theta = data.theta

#Calculate the geostrophic wind and rotate the vectors to the east
 #rotate geostrophic wind vectors to the east
g = 9.81
R = 287.
f = 2 * 7.29e-5 * np.sin(gw_data.coords_cabauw[1] * np.pi/180.)

def f_dpdz_dTdz(y): #y = np.array([p, T])
    p, T = y
    dpdz = -g/R * p/T
    return np.array([dpdz, dTdz]) #dTdz is defined in the loop below
    
p_0 = data.p0 * 100. #Pressure at 1.5 m
p_200m = p_0.copy()
for i in range(1, len(data.z)):
    z_old = data.z[-i]; z_new = data.z[-(i+1)]
    dz = z_new - z_old
    dTdz = (data.T[:,:,-(i+1)] - data.T[:,:,-i]) / dz
    p_200m += ft.RK4(f_dpdz_dTdz, np.array([p_200m, data.T[:,:,-i]]), dz)[0]
V_thermal = R / f * np.array([[[-1, 1]]]) * tg_data.T_gradient_xy[:,:,::-1] * np.log(p_200m / p_0)[:,:,np.newaxis]

#%%
# Find the geostrophic wind, rotate the vectors to the east and normalize the wind vectors with the geostrophic wind
Vg = gw_data.V_g[:,:,np.newaxis,:] + (data.z[np.newaxis, np.newaxis, :-1, np.newaxis] - 1.5) / 198.5 * V_thermal[:,:,np.newaxis,:]
Vg_speed = np.linalg.norm(Vg, axis = 3)
Vg_speed_0 = Vg_speed[:,:,-1]
V = data.V[:,:,:-1]
 
angle_rotate = np.arctan2(Vg[:,:,:,1], Vg[:,:,:,-1])
rot_matrix = np.zeros(V.shape[:3] + (2, 2))
rot_matrix[:,:,:,0,0] = np.cos(angle_rotate); rot_matrix[:,:,:,0,1] = np.sin(angle_rotate)
rot_matrix[:,:,:,1,0] = - np.sin(angle_rotate); rot_matrix[:,:,:,1,1] = np.cos(angle_rotate)
normalized_V = np.matmul(V[:,:,:,np.newaxis,:], np.transpose(rot_matrix, axes = [0,1,2,4,3]))[:,:,:,0] / Vg_speed[:,:,:,np.newaxis]

#Vg_speed_0 = Vg_speed_0.flatten()
#normalized_V = np.reshape(normalized_V, (V.shape[0] * V.shape[1], V.shape[2], V.shape[3]))
dtheta = (data.theta[:,:,0] - data.theta[:,:,-2])#.flatten() #Difference in theta between 10 and 200 m

dtheta_criterion = (dtheta[:,0] < -1) & (dtheta[:, 72] > 3)
Vg_hours = Vg[:, ::6] 


#stability_classes = [[-2, 0]]
#data_classes = {}
#data_means_classes = {}
#for c in stability_classes:
#    data_classes[str(c)] = {}; data_means_classes[str(c)] = {}
#    in_class = (dtheta >= c[0]) & (dtheta < c[1]) & (Vg_speed_0 <= 7.5)
#    print(c, np.count_nonzero(in_class))
#    data_classes[str(c)]['Vg_speed_0'] = Vg_speed_0[in_class]
#    data_classes[str(c)]['normalized_V'] = normalized_V[in_class]
#    data_classes[str(c)]['dtheta'] = dtheta[in_class]
#    
#    data_means_classes[str(c)]['Vg_speed_0'] = np.mean(data_classes[str(c)]['Vg_speed_0'])
#    data_means_classes[str(c)]['normalized_V'] = np.mean(data_classes[str(c)]['normalized_V'], axis = 0)
#    data_means_classes[str(c)]['dtheta'] = np.mean(data_classes[str(c)]['dtheta'])  
#    
#plt.figure()
#for c in stability_classes:
#    c = str(c)
#    plt.plot(data_means_classes[c]['normalized_V'][:,0], data_means_classes[c]['normalized_V'][:,1])
#    plt.xlabel('normalized u (m/s)', fontsize = 12); plt.ylabel('normalized y (m/s)', fontsize = 12)
#plt.show()
#%%

#%%
u200_normalized = normalized_V[:,:,0,0]
v200_normalized = normalized_V[:,:,0,1]
V200_normalized = normalized_V[:,:,0,:]

V200_normalized_jan = normalized_V[1:32,:,0,:]

u200_mean_jan = np.mean(u200_normalized[1:32,:])
v200_mean_jan = np.mean(v200_normalized[1:32,:])


plt.figure(figsize = (6,6))
for i in range(0,31):
        plt.plot(V200_normalized_jan[i,:,0], V200_normalized_jan[i,:,1])
#plt.axes().set_aspect('equal', 'box')
plt.show

theta_unstable_mask = (dtheta < 0)
theta_unstable = dtheta[theta_unstable_mask]
V200_normalized_unstable = V200_normalized[theta_unstable_mask] 

Vg200_speed = Vg_speed[:,:,0]
Vg_mask = (Vg200_speed < 5)
Vg200_speed_limit = Vg200_speed[Vg_mask]
V200_normalized_unstable_geolimit = V200_normalized_unstable[Vg_mask] 

plt.figure(figsize = (6,6))
plt.plot(V200_normalized_unstable[:,0], V200_normalized_unstable[:,1])
plt.show()

#Vg_speed_unstable = Vg_speed[theta_unstable_mask]





        
