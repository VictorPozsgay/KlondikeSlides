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

from KlondikeSlides.constants import holocene_end_kaBP


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

def format_simulations_ygs(path_database, path_in, path_out):
    """ Function formats simulation data into a netCDF pickle
    
    Parameters
    ----------
    path_database : str
        Path to the folder where all the raw GST logger data are found
    path_in : str
        Path to the folder where the raw simulation data is found
    path_out : str
        Path to the processed output pickle
    """
    list_subsites_db = []
    list_sites_db = []

    for f in os.listdir(path_database):
        if 'hourly' in f:
            file = f.split('_hourly')[0]
            list_subsites_db.append(file.split('-')[1])
            list_sites_db.append(file.split('-')[1].split('_ST')[0])

    list_subsites_db = sorted(list_subsites_db)
    list_sites_db = list(np.unique(list_sites_db))

    list_reanalysis = sorted(next(os.walk(path_in))[1])

    metadata_sims_csv = {ra: [] for ra in list_reanalysis}

    for ra in list_reanalysis:
        metadata_sims_csv[ra] = pd.read_csv(os.path.join(path_in,f'{ra}/metadata.csv'))
        metadata_sims_csv[ra] = metadata_sims_csv[ra].drop(columns=['id', 'model', 'parameters', 'site_name', 'forcing_name'])
        metadata_sims_csv[ra]['site'] = [i.split('YGS_')[1] for i in metadata_sims_csv[ra]['site']]
        metadata_sims_csv[ra]['forcing'] = [i.split('_stations')[0] for i in metadata_sims_csv[ra]['forcing']]
        metadata_sims_csv[ra]['directory'] = [i.split('_')[-1] for i in metadata_sims_csv[ra]['directory']]
        for var in ['topo', 'loc', 'forest', 'soil']:
            metadata_sims_csv[ra][var] = [i.split(f'{var}_')[1] for i in metadata_sims_csv[ra][var]]
        metadata_sims_csv[ra]['suffix'] = [metadata_sims_csv[ra]['topo'][i].split(loc)[1] for i,loc in enumerate(metadata_sims_csv[ra]['loc'])]
        metadata_sims_csv[ra]['sub_site'] = [{'_a': 'ST01', '_b': 'ST02', '_c': 'ST03', '': ''}[i] for i in metadata_sims_csv[ra]['suffix']]
        metadata_sims_csv[ra]['site_and_sub'] = [i+('' if j=='' else '_') +j for i,j in zip(metadata_sims_csv[ra]['site'], metadata_sims_csv[ra]['sub_site'])]
        
    list_sites_including_sub = sorted(np.unique(metadata_sims_csv[list_reanalysis[0]]['site_and_sub']))

    # Here we compute the 'calibrated mean' as
    # The mean of all simulations with 'mean SnowCorrFactor',
    # 'mean SnowViscosity'

    nc_soil = {ra: Dataset(os.path.join(path_in,f'{ra}/result_soil_temperature.nc'))['geotop'] for ra in list_reanalysis}

    time_soil = pd.to_datetime([str(i) for i in num2date(nc_soil[list_reanalysis[0]]['Date'][:], nc_soil[list_reanalysis[0]]['Date'].units)])

    df_Tg = {ra: pd.DataFrame(nc_soil[ra]['Tg'][:,:,0].T,
                            index=time_soil,
                            columns=[i.split('_')[-1] for i in nc_soil[ra]['simulation'][:]])
            for ra in list_reanalysis}

    df_Tg_sites_including_subs = {site: {ra: [] for ra in list_reanalysis} for site in list_sites_including_sub}
    df_Tg_sites = {site: {ra: [] for ra in list_reanalysis} for site in list_sites_db}

    for site in list_sites_db:
        for ra in list_reanalysis:
            df_meta = metadata_sims_csv[ra].loc[(metadata_sims_csv[ra]['site']==site)]
            list_subs = sorted(np.unique(df_meta['sub_site']))
            for sub in list_subs:
                site_with_sub = site + ('' if sub=='' else f'_{sub}')
                df_meta_sub = df_meta.loc[(df_meta['sub_site'])==sub]

                list_scf = sorted(np.unique(df_meta_sub['SnowCorrFactor']))
                list_visc = sorted(np.unique(df_meta_sub['SnowViscosity']))
                all_dirs = list(df_meta_sub.directory)
                calibrated_dir = list(df_meta_sub.loc[(df_meta_sub['SnowCorrFactor']==list_scf[1]) & (df_meta_sub['SnowViscosity']==list_visc[1]), 'directory'])
                df_Tg_sites_including_subs[site_with_sub][ra] = df_Tg[ra][all_dirs].copy()
                df_Tg_sites_including_subs[site_with_sub][ra]['GST_sim_min_C'] = df_Tg_sites_including_subs[site_with_sub][ra].min(axis=1)
                df_Tg_sites_including_subs[site_with_sub][ra]['GST_sim_max_C'] = df_Tg_sites_including_subs[site_with_sub][ra].max(axis=1)
                df_Tg_sites_including_subs[site_with_sub][ra]['GST_sim_calibrated_C'] = df_Tg_sites_including_subs[site_with_sub][ra][calibrated_dir].mean(axis=1)

    for site in list_sites_db:
        for ra in list_reanalysis:
            if site in list_sites_including_sub:
                df_Tg_sites[site][ra] = df_Tg_sites_including_subs[site][ra].copy()
            else:
                df_Tg_sites[site][ra] = pd.concat([df_Tg_sites_including_subs[f'{site}_ST0{i}'][ra].drop(columns=['GST_sim_min_C', 'GST_sim_max_C', 'GST_sim_calibrated_C']) for i in range(1,4)],
                                                join='inner', axis=1)
                df_Tg_sites[site][ra]['GST_sim_min_C'] = pd.concat([df_Tg_sites_including_subs[f'{site}_ST0{i}'][ra][['GST_sim_min_C']] for i in range(1,4)],
                                                        join='inner', axis=1).min(axis=1)
                df_Tg_sites[site][ra]['GST_sim_max_C'] = pd.concat([df_Tg_sites_including_subs[f'{site}_ST0{i}'][ra][['GST_sim_max_C']] for i in range(1,4)],
                                                        join='inner', axis=1).max(axis=1)
                df_Tg_sites[site][ra]['GST_sim_calibrated_C'] = pd.concat([df_Tg_sites_including_subs[f'{site}_ST0{i}'][ra][['GST_sim_calibrated_C']] for i in range(1,4)],
                                                        join='inner', axis=1).mean(axis=1)

    df_Tg_sites_summary = {site: {ra: [] for ra in list_reanalysis} for site in list_sites_db}
    for site in list_sites_db:
        for ra in list_reanalysis:
            df_Tg_sites_summary[site][ra] = df_Tg_sites[site][ra][['GST_sim_min_C', 'GST_sim_max_C', 'GST_sim_calibrated_C']]

    # Open a file and use dump() 
    with open(os.path.join(path_out, 'df_Tg_sites_summary.pkl'), 'wb') as file: 
        # A new file will be created 
        pickle.dump(df_Tg_sites_summary, file)

def format_paleo_Benthic(path_in, path_out):
    """ Function formats Benthic paleo data into a csv
    
    Parameters
    ----------
    path_in : str
        Path to the raw input .csv file where the Benthic paleo data is stored
    path_out : str
        Path to the processed output .csv file where the Benthic paleo data is stored  
    """
    df = pd.read_csv(path_in, comment='#', sep='\t')
    df = df.rename(columns={k:v for k,v in zip(df.columns, ['age_ka', 'Benthic_d18O', 'se'])})
    df = df.set_index('age_ka')[['Benthic_d18O']]
    df.to_csv(path_out)

def format_paleo_Greenland(path_in, path_out):
    """ Function formats Greenland paleo data into a csv
    
    Parameters
    ----------
    path_in : str
        Path to the raw input .csv file where the Greenland paleo data is stored
    path_out : str
        Path to the processed output .csv file where the Greenland paleo data is stored  
    """
    df = pd.read_csv(path_in, comment='#', sep='\t')
    df['age_ka'] = pd.to_numeric(df["age_calBP1950"]) / 1000.0
    df = df[['age_ka','d18O_smow']].rename(columns={'d18O_smow': 'Greenland_d18O'})
    df = df.set_index('age_ka')
    window=5
    df['forward'] = df['Greenland_d18O'][::-1].rolling(window=window).mean()[::-1]
    df['backward'] = df['Greenland_d18O'].rolling(window=window).mean()
    df['Greenland_d18O'] = df[['forward','backward']].mean(skipna=True,axis=1)
    df = df.loc[df.index<holocene_end_kaBP,['Greenland_d18O']]
    df.to_csv(path_out)

def format_paleo_Yukon(path_in, path_out):
    """ Function formats Yukon paleo data into a csv
    
    Parameters
    ----------
    path_in : str
        Path to the raw input .csv file where the Yukon paleo data is stored
    path_out : str
        Path to the processed output .csv file where the Yukon paleo data is stored  
    """
    df = pd.read_csv(path_in)
    df['age_ka'] = df['Year_before_1950']/1000
    df = df.set_index('age_ka')[['dT', 'dT_minus_1s', 'dT_plus_1s']]
    df = df.loc[df.index<holocene_end_kaBP]
    df.to_csv(path_out)

def format_all_raw_data(path_wildfire_in, path_wildfire_out, path_gst_logger_in, path_gst_logger_out,
                        path_forcing_ygs_sites_in, path_forcing_ygs_sites_out, path_simulations_ygs_in, path_simulations_ygs_out,
                        path_paleo_Benthic_in, path_paleo_Benthic_out, path_paleo_Greenland_in, path_paleo_Greenland_out, path_paleo_Yukon_in, path_paleo_Yukon_out):
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
    path_simulations_ygs_in : str
        Path to the folder where the raw simulation data is found
    path_simulations_ygs_out : str
        Path to the processed output pickle
    path_paleo_Benthic_in : str
        Path to the raw input .csv file where the Benthic paleo data is stored
    path_paleo_Benthic_out : str
        Path to the processed output .csv file where the Benthic paleo data is stored 
    path_paleo_Greenland_in : str
        Path to the raw input .csv file where the Greenland paleo data is stored
    path_paleo_Greenland_out : str
        Path to the processed output .csv file where the Greenland paleo data is stored  
    path_paleo_Yukon_in : str
        Path to the raw input .csv file where the Yukon paleo data is stored
    path_paleo_Yukon_out : str
        Path to the processed output .csv file where the Yukon paleo data is stored 

    Returns
    -------
    
    """
    format_wildfire_data(path_wildfire_in, path_wildfire_out)
    format_gst_logger(path_gst_logger_in, path_gst_logger_out)
    format_forcing_ygs(path_forcing_ygs_sites_in, path_forcing_ygs_sites_out)
    format_simulations_ygs(path_gst_logger_in, path_simulations_ygs_in, path_simulations_ygs_out)
    format_paleo_Benthic(path_paleo_Benthic_in, path_paleo_Benthic_out)
    format_paleo_Greenland(path_paleo_Greenland_in, path_paleo_Greenland_out)
    format_paleo_Yukon(path_paleo_Yukon_in, path_paleo_Yukon_out)
