# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 11:25:27 2018

@author: bramv
"""
import calendar
import requests
import os
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import settings as s
import functions as ft



class TemperatureGradient():
    def __init__(self):
        pass
    
    

def calculate_temperature_gradient(years = [], months = []):
    """This function calculates the geostrophic wind at and elevation of about 10 m using pressure observations, and returns 
    a data object that contains the relevant datasets as attributes.
    If year and months are unspecified, then the year and month specified in settings.py are used.
    Otherwise months should be a list with the months that need to be considered.
    """
    if len(months) == 0: months = [s.month]
    else: months = [j if isinstance(j, str) else format(j, '02d') for j in months]
    if len(years) == 0: years = [s.year for j in months]
    else: years = [str(j) for j in years]
    n_days = [calendar.monthrange(int(years[k]), int(months[k]))[1] for k in range(len(months))]
    
    tg_data = TemperatureGradient()
    
    
    
    def model(coords, dTdphi, dTdlambda):
        return T_0 + dTdphi * (coords[:,0] - phi_0) + dTdlambda * (coords[:,1] - lambda_0)
    
    for k in range(len(months)):
        print(months[k])
        filename = s.data_path+'KNMI_'+years[k]+months[k]+'_hourly_temperature.txt'
        
        if not os.path.exists(filename):
            url = 'http://projects.knmi.nl/klimatologie/uurgegevens/getdata_uur.cgi'
            r = requests.post(url, data={"stns":'ALL', "start":years[k]+months[k]+"0101", "end":years[k]+months[k]+str(n_days[k])+"24", "vars":"T"})
            content = r.text
            with open(filename, 'w') as f:
                f.write(content)

        
        
        data = np.char.strip(np.loadtxt(filename, dtype='str', delimiter = ','))
        
        """Check for stations with incomplete datasets, and remove these stations from the list"""
        hours = data[:, 2].astype('int')
        diff1 = hours[1:] - hours[:-1]
        test1 = (diff1 > 1) | ((diff1 < 1) & (diff1 > -23))
        dates = data[:, 1].astype('int')
        diff2 = dates[1:] - dates[:-1]
        test2 = diff2 > 1 
        if np.count_nonzero(test1) > 0 or np.count_nonzero(test2) > 0:
            data1 = data[1:][test1]
            data2 = data[1:][test2]
            stations_remove = np.unique(np.append(data1[:,0], data2[:,0]))
            data = data[np.logical_and.reduce([data[:,0] != j for j in stations_remove])]
        
        stations = data[:,0].astype('int')
        #The stations might not be sorted, and np.unique returns a sorted array by default, which is
        #not desired here. With the method below this is circumvented.
        stations_unique_indices = np.unique(stations, return_index = True)[1]
        stations_unique = stations[np.sort(stations_unique_indices)] 
        temperatures = data[:,3]
        temperatures[temperatures == ''] = np.nan
        temperatures = temperatures.astype('float') / 10.
        
        #Reshape the pressure array, in such a way that observations for different stations and dates are grouped.
        index = np.where(np.abs(stations[1:] - stations[:-1]) > 0)[0][0] + 1
        n_times = 24
        n_dates = int(index / 24)
        n_stations = int(len(data) / (n_dates * n_times))
        temperatures = np.reshape(temperatures, (n_stations, n_dates, n_times))
        temperatures = {stations_unique[j] : temperatures[j] for j in range(n_stations)}
        #print(list(temperatures.keys()))
        
        #Read coordinates of the stations
        with open(filename,'r') as f:
            text = f.read()
            i1 = text.index('NAME')+6
            i2 = text.index('YYYYMMDD')-11
            
            coords_data = ft.list_data(text[i1:i2+5], separator = ' ')
            coords = {int(j[1][:-1]) : np.array([j[2], j[3]], dtype = 'float') for j in coords_data}
        
        cabauw_id = int([j[1][:-1] for j in coords_data if j[-1] == 'CABAUW'][0])
        
        distances_to_cabauw = {j: ft.haversine(coords[cabauw_id][0], coords[cabauw_id][1], coords[j][0], coords[j][1]) for j in coords if not j == cabauw_id}
        elevations = {int(j[1][:-1]) : float(j[4]) for j in coords_data}
        max_distance = 75 #km
        max_elevation = 15 #In m
        selected_stations = [j for j in distances_to_cabauw if distances_to_cabauw[j] < max_distance and elevations[j] < max_elevation and j in temperatures and not np.isnan(temperatures[j][0][0])]
        #print('selected_stations: ',[j[-1] for j in coords_data if int(j[1][:-1]) in selected_stations and distances_to_cabauw[int(j[1][:-1])] < 75])
        
        selected_temperatures = np.zeros((n_dates, n_times, len(selected_stations)))
        for j in range(len(selected_stations)):
            selected_temperatures[:, :, j] = temperatures[selected_stations[j]]
            
        selected_coords = np.array([coords[j] for j in selected_stations])[:,::-1] * np.pi/180. #Swap latitude and longitude
                    
        R = 6371e3
        T_gradient = np.zeros((n_dates, n_times, 2))
        for i in range(0, n_dates):
            for j in range(0, n_times):
                T_0 = temperatures[cabauw_id][i][j]
                phi_0 = coords[cabauw_id][1] * np.pi/180.
                lambda_0 = coords[cabauw_id][0] * np.pi/180.
                estimates = curve_fit(model, selected_coords, selected_temperatures[i, j], [0,0])
                T_gradient[i, j] = estimates[0]
          
        T_gradient_xy = 1./ R * T_gradient * np.array([[[-1., 1./np.cos(coords[cabauw_id][1] * np.pi/180.)]]])
        #*100 is to convert temperatures from hPa to Pa
        """T_gradient_xy is now available every hour, from hour 1 to 24, while observations at Cabauw are available every 10 minutes.
        T_gradient_xy therefore needs to be interpolated to the times at which Cabauw observations are available.
        This is done by taking for each Cabauw observation the pressure at the nearest hour.
        """
        T_gradient_xy_interpolated = np.zeros((T_gradient_xy.shape[0], T_gradient_xy.shape[1] * 6, 2))
        T_gradient_xy_interpolated[:, :3] = T_gradient_xy[:,np.newaxis, 0]; T_gradient_xy_interpolated[:, -3:] = T_gradient_xy[:,np.newaxis, -1]
        
        for i in range(0, T_gradient_xy.shape[1] - 1):
            T_gradient_xy_interpolated[:, 3 + 6*i : 9 + 6*i] = T_gradient_xy[:, np.newaxis, i]
        T_gradient_xy = T_gradient_xy_interpolated
            
        variables = ['T_gradient_xy']
        for j in variables:
            if k == 0:
                exec('tg_data.'+j+' = '+j)
            else:
                exec('tg_data.'+j+' = np.concatenate([tg_data.'+j+','+j+'], axis = 0)')
    return tg_data


if __name__ == '__main__':
    #This part is not executed when importing this script, only when running this as the main script.
    fig, ax = plt.subplots(1, 1)
    
    tg_data = calculate_temperature_gradient()

    ax.plot(tg_data.T_gradient_xy[:,:,0], tg_data.T_gradient_xy[:,:,1], 'bo')
    plt.show()