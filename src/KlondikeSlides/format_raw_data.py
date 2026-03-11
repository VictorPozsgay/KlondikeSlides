"""This module formats raw observation data into processed data"""

#pylint: disable=line-too-long
#pylint: disable=trailing-whitespace
#pylint: disable=invalid-name

import pandas as pd
from netCDF4 import Dataset, num2date #pylint: disable=no-name-in-module
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from datetime import datetime, timedelta
import os
import pickle
from functools import reduce


def format_wildfire_data(path_in, path_out):
    """ Function formats wildfire data into a csv
    
    Parameters
    ----------
    path_in : str
        Path to the raw input .csv file where the wildfire data is stored
    path_out : str
        Path to the processed output .csv file where the wildfire data is stored  
    """
    df = pd.read_csv(path_in)[['Fire Year', 'SHAPE_Area (within inventory region only)']]
    df = df.rename(columns={'Fire Year':'Year','SHAPE_Area (within inventory region only)':'area_kha'})
    df['area_kha'] = df['area_kha']/1000/100**2
    df = df.groupby('Year').sum()
    df = df.loc[df.index<=2026]
    df.index = pd.to_datetime(df.index,format='%Y')
    df = df.resample('YE').mean().fillna(0)
    df.index = df.index.year
    df.to_csv(path_out)

def format_gst_logger(path_in, path_out):
    """ Function formats GST logger data into a netCDF pickle
    
    Parameters
    ----------
    path_in : str
        Path to the folder where all the raw GST logger data are found
    path_out : str
        Path to the processed output pickle
    """
    list_subsites_db = []
    list_sites_db = []

    for f in os.listdir(path_in):
        file = f.split('_hourly')[0]
        list_subsites_db.append(file.split('-')[1])
        list_sites_db.append(file.split('-')[1].split('_ST')[0])

    list_subsites_db = sorted(list_subsites_db)
    list_sites_db = list(np.unique(list_sites_db))

    dic_gst_obs = {site: {freq: [] for freq in ['hourly', 'daily']} for site in list_sites_db}

    for site in list_sites_db:
        list_subs = [i.split('_')[-1] for i in list_subsites_db if site in i]
        local_dic = {sub: [] for sub in list_subs}
        for sub in list_subs:
            path_file = f'{path_in}/YGS-{site}_{sub}_hourly_2025.nc'
            if os.path.isfile(path_file):
                df = Dataset(path_file, mode='r')
                local_dic[sub] = pd.DataFrame(np.array([num2date(df['time'], df['time'].units)]).T, columns=['time'])
                local_dic[sub]['Date'] = pd.to_datetime([str(i) for i in local_dic[sub]['time']])#.strftime('%Y-%m-%d %H:%M:%S')
                local_dic[sub] = local_dic[sub].drop(columns=['time'])
                local_dic[sub][f'GST_obs_{sub}_C'] = df['ground_temperature'][0,:,0]
        dic_gst_obs[site]['hourly'] = reduce(lambda  left,right: pd.merge(left,right,on=['Date'], how='outer'), [i for i in local_dic.values() if type(i) != list]).set_index('Date')
        dic_gst_obs[site]['daily'] = dic_gst_obs[site]['hourly'].resample('1D').mean()

    for site in list_sites_db:
        for freq in ['hourly', 'daily']:
            dic_gst_obs[site][freq].columns = pd.MultiIndex.from_product([['raw'], dic_gst_obs[site][freq].columns])
            dic_gst_obs[site][freq][('stats', 'GST_obs_min_C')] = dic_gst_obs[site][freq]['raw'].min(axis=1)
            dic_gst_obs[site][freq][('stats', 'GST_obs_mean_C')] = dic_gst_obs[site][freq]['raw'].mean(axis=1)
            dic_gst_obs[site][freq][('stats', 'GST_obs_max_C')] = dic_gst_obs[site][freq]['raw'].max(axis=1)

    # Open a file and use dump() 
    with open(os.path.join(path_out, 'dic_gst_obs.pkl'), 'wb') as file: 
        # A new file will be created 
        pickle.dump(dic_gst_obs, file)

def format_forcing_ygs(path_in, path_out):
    """ Function formats forcing data into a netCDF pickle
    
    Parameters
    ----------
    path_in : str
        Path to the folder where the raw forcing data is found
    path_out : str
        Path to the processed output pickle
    """
    list_reanalysis = []
    for f in os.listdir(path_in):
        if '_ygs_sites_crop.nc' in f:
            list_reanalysis.append(f.split('_ygs')[0]) 

    dic_forcings = {ra: os.path.join(path_in, f'{ra}_ygs_sites_crop.nc') for ra in list_reanalysis}
    list_stations = [i.decode("utf-8").split('YGS_')[-1] for i in xr.open_dataset(dic_forcings[list_reanalysis[0]], engine="netcdf4")['station_name'].values]

    df_airT_dic = {ra: xr.open_dataset(dic_forcings[ra], engine="netcdf4")['AIRT_pl'].to_dataframe().unstack() for ra in list_reanalysis}
    for ra in list_reanalysis:
        df_airT_dic[ra].columns = pd.MultiIndex.from_product([[ra], list_stations], names=['reanalysis', 'station'])
    
    df_airT_all = pd.concat([df_airT_dic[ra] for ra in list_reanalysis], axis=1)
    df_airT = df_airT_all.swaplevel(axis=1)[pd.MultiIndex.from_product([list_stations, list_reanalysis], names=['station', 'reanalysis'])]

    # Open a file and use dump() 
    with open(os.path.join(path_out, 'df_airT.pkl'), 'wb') as file: 
        # A new file will be created 
        pickle.dump(df_airT, file)
        

def format_all_raw_data(path_wildfire_in, path_wildfire_out, path_gst_logger_in, path_gst_logger_out, path_forcing_ygs_sites_in, path_forcing_ygs_sites_out):
    """ Function formats all raw input data
    
    Parameters
    ----------
    path_wildfire_in : str
        Path to the raw input .csv file where the wildfire data is stored
    path_wildfire_out : str
        Path to the processed output .csv file where the wildfire data is stored
    path_gst_logger_in : str
        Path to the folder where all the raw GST logger data are found
    path_gst_logger_out : str
        Path to the processed output pickle
    path_forcing_ygs_sites_in : str
        Path to the folder where the raw forcing data is found
    path_forcing_ygs_sites_out : str
        Path to the processed output pickle

    Returns
    -------
    
    """
    format_wildfire_data(path_wildfire_in, path_wildfire_out)
    format_gst_logger(path_gst_logger_in, path_gst_logger_out)
    format_forcing_ygs(path_forcing_ygs_sites_in, path_forcing_ygs_sites_out)
