import os
import sys
import json
import requests
import pandas as pd
import numpy as np
from random import sample
import random
import fiona
import matplotlib.pyplot as plt
from datetime import datetime
from database.db import tables, execute_query, config
from database import db_queries
from database.utils import keyword_search

pd.options.mode.chained_assignment = None  # default='warn'
kywd = keyword_search(os.path.join(os.getcwd(), 'config/config_file.txt'), ['DS_config'])
path_conf = os.path.dirname(kywd['DS_config'])
conf = os.path.basename(kywd['DS_config'])
conf_file = os.path.join(os.getcwd(), path_conf, conf)


def parcelTimeSeriesDfLocal(dataset='test_2021', pidList=None, stat=None, bands=None, geom_column='geometry'):

    if stat is None:
        stat = ['mean', 'std', 'min', 'p25', 'median', 'p75', 'max']

    stat.extend(['count', 'systemindex'])
    # get info from Dataset JSON - TODO: include automatic writing of dataset JSON in sen4cbm
    ds_json = config.read(conf_file)[dataset]
    parcel_id = ds_json['pcolumns']['parcel_id']
    if pidList is None:
        getParcelId = f"SELECT DISTINCT {parcel_id} from {ds_json['tables']['parcels']}"
        rows = execute_query(getParcelId, db=ds_json['db'])
        pidList = [str(r[0]) for r in rows]

    # Verify input
    if type(pidList) == str:
        pidList = list(pidList.strip().split(','))
    elif type(pidList) == int:
        pidList = list(str(pidList))
    # Get tables names based on dataset and bands needed
    tables_list = tables(db=ds_json['db'], matching_text=dataset, all_records=True)
    if bands is not None:
        match_list = list()
        for band in bands:
            match = [t for t in tables_list if '_' + band.lower() in t]
            match_list.append(match[0])
    else:
        match_list = tables_list
    # Retrieve parcel data
    parcel_data = None
    pidList = [p.strip() for p in pidList]
    #Modified to allow multiple parcels
    appo = db_queries.getParcelById(ds_json, pidList, geom_column=geom_column)
    parcel_data = pd.DataFrame(columns=appo[0])
    for i in range(1, len(appo)):
        parcel_data = parcel_data.append(pd.Series(appo[i], index=parcel_data.columns), ignore_index=True)
    # for p in pidList:
    #     appo = db_queries.getParcelById(ds_json, p, geom_column=geom_column)
    #     if parcel_data is None:
    #         parcel_data = pd.DataFrame(columns=appo[0])
    #     if len(appo) > 1:
    #         parcel_data = parcel_data.append(pd.Series(appo[1], index=parcel_data.columns), ignore_index=True)
    # Retrieve time series data
    full_data = pd.DataFrame()
    pidList = [f"'{p}'" for p in pidList]
    for tbl in match_list:
        data = pd.DataFrame(columns=['pid', 'Date']+stat)
        if tbl.find('grd_') != -1:
            band = tbl[tbl.find('grd_') + 4:]
        else:
            band = tbl[tbl.find('boa_') + 4:]
        metrics = ', '.join(stat)
        getTableDataSql = f"SELECT {parcel_id} as pid, acqdate, {metrics}" \
                          f" FROM {tbl}" \
                          f" WHERE {parcel_id} IN ({', '.join(pidList)});"
        rows = execute_query(getTableDataSql, db=ds_json['db'])
        if len(rows) > 0:
            for r in rows:
                data = data.append(pd.Series(r, index=data.columns), ignore_index=True)
        data['band'] = band
        if any(p.replace("'",'').isnumeric() for p in pidList) is True:
            appo = (parcel_data.set_index(parcel_data['pid'].astype('float')))['cropname']
        else:
            appo = (parcel_data.set_index(parcel_data['pid'].astype('str')))['cropname']
        df = data.set_index('pid').join(appo, how='inner', rsuffix='_joined')
        full_data = full_data.append(df)
    return full_data.reset_index()

def parcelTimeSeriesDfOutreach(host, username, password, pidList, year, tstype):
    """
    This function returns the parcel time series data as a dataframe for one or more input parcels. It returns this data
    from the JRC Outreach database for IE using the JRC RESTful API.
    
    Parameters:
    
    - host: String - IP address for Restful access
    - username: String - username for Restful services
    - password: String - password for Restful services
    - pidList: String or List - can be a string (single parcel id) or a list of strings (multiple parcels)    
    - year: Integer
    - tstype: String - Must be one of the following: 's2', 'bs', 'c6'

    """

    # If the pidList is a string with a single parcel, convert it to a list:
    if type(pidList) == str:
        pidList = list(pidList.split(','))

    # loop through each parcel ID in the pidList.
    # First, send a 'parcelById' request via the Restful API. Extract the unique crop type for the given parcel.
    # Next, send a 'parcelTimeSeries' request via the Restful API. Convert the json response to a dataframe and 
    # add a 'pid' and 'crop' column.
    # Finally, for each parcel in the list, append the resulting dataframe onto the master dataframe.
    df = None
    for pid in pidList:
        pidurl = """http://{}/query/parcelByID?aoi=ie&year={}&pid={}"""
        pidResponse = requests.get(pidurl.format(host, year, pid), auth=(username, password))
        crop = json.loads(pidResponse.content)['cropname'][0]
        tsurl = """http://{}/query/parcelTimeSeries?aoi=ie&year={}&pid={}&tstype={}"""
        tsResponse = requests.get(tsurl.format(host, year, pid, tstype), auth=(username, password))
        if df is None:
            df = pd.read_json(tsResponse.content)
            if not df.empty:
                df['pid'] = pid
                df['crop'] = crop
        else:
            df1 = pd.read_json(tsResponse.content)
            if not df1.empty:
                df1['pid'] = pid
                df1['crop'] = crop
                df = df.append(df1)

    if df.empty:
        print('Timeseries query returned empty {} result for parcel {}'.format(tstype, pid))
        sys.exit()

    else:
        return df



def plotTimeSeriesLocal(df, stat, dataset='test_2021', bands=None, start_date=None, end_date=None, freq_avg=False, frequency='W',
                        window_size=None, orbit=None, legend=True, class_colours=False, logs=False,
                        set_ylim=False, ymin=None, ymax=None):
    
    """
    This function creates time series graphs for the given parcel(s). Graphs will be plotted for any of the 
    following card types depending on the input dataframe: 's2', 'bs', 'c6'.
    The user can also determine whether to apply a rolling average to the data or a weekly average if desired.
    Additionally, the user can choose to plot DESC and ASC data separately or together.

    Parameters:

    - df: Dataframe - Dataframe with time series data for selected parcels
    - stat: String - The user can select from the following: 'mean', 'std', 'min', 'max',
            'p25', 'p50', 'p75'
    - dataset: String - Name of the dataset as defined in datasets.json config file
    - bands: String or List - Options are 'blue','green','nir','re1','re2','re3','re4','red','swir1','swir2',
            'nbr','nbr2','ndre','ndvi','reivi' for s2, 'vv','vh','vv-vh','vvonvh','vhonvv' for bs, 'vv','vh' 
            for c6. 
            Default if none selected is 'NDVI' for s2 and 'VV' and 'VH' for bs/c6.
    - start_date: String - format must be '%Y-%m-%d'. Default if none specified is date of first entry
            in dataframe
    - end_date: String - format must be '%Y-%m-%d'. Default if none specified is date of last entry
            in dataframe
    - freq_avg: Boolean - User can apply a frequency average to the data. Default is False
    - frequency: String - If freq_avg is True, user can apply a specific frequency to average the data.
            Examples are 'W', '2W', 'M', '6M', '3D' etc. Default is 'W'
    - window_size: Integer - User can apply a moving average smoothing function to the data with a 
            specified window size. Default is no moving average smoothing function
    - orbit: String - User can define whether to plot DESC and ASC data merged or whether to plot just DESC data
            of just ASC data. Options are None (ASC and DESC together), 'A', 'D'. Default is None
    - legend: Boolean - User can determine whether to plot a legend with parcel number and crop type or not.
            Default is True
    - class_colours: Boolean - User can decide if they wish to plot all input time series with a different colour
            for each or all input time series of the same class (crop type) with the same colour.
            Default is False
    - logs: Boolean - True or False: The user can also decide to plot log activities if they exist.
            Default is False
    - set_ylim: Boolena - User can decide to set ymin and ymax values for the y axis.
            Default is False
    - ymin: Float - Minimum value for Y axis if required.
            Default is None
    - ymax: Float - Maximum value for Y axis if required
            Default is None

    """
    

    # Input dataframe refinements:
    try:

        df = df.copy(deep=True)

        # Determine the orbit direction (Descending or Ascending) for S1:
        df['satellite'] = df['systemindex'].apply(lambda s: 'S1' if s[:2] == 'S1' else 'S2')

        df_index = df.index

        dfs1_index = df_index[df['satellite'] == 'S1']

        df['orbit'] = 'NAN'
        if len(dfs1_index) > 0:
            ref_index = df['systemindex'].iloc[dfs1_index[0]].find('DV')
            ref_index = ref_index + 12
            # Create an 'orbit' column with information about the Asc and Desc orbits:
            df['orbit'].iloc[dfs1_index] = df['systemindex'].iloc[dfs1_index].apply(lambda s: 'D' if int(s[ref_index:ref_index+2]) < 12 else 'A')

        df = df.drop(['systemindex'], axis=1)


        # Add 'grd_' to the band name of any S1 GRD time series:
        grd_list = ['vh','vv','vhonvv','vv_vh','vvonvh']
        grd_index = df_index[df['band'].isin(grd_list)]
        df['band'].iloc[grd_index] = ['grd_' + item for item in df['band'].iloc[grd_index]]


        # Convert the datetime.date column to datetime.datetime:
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')  

        # Get subset of dataframe if start_date and end_date specified:
        if start_date:
            df = df[df['Date'] >= datetime.strptime(start_date, '%Y%m%d')]

        if end_date:
            df = df[df['Date'] <= datetime.strptime(end_date, '%Y%m%d')]

        # Return list of parcels:
        pid = df['pid'].unique()

        # Return list of unique cropnames for selected parcels:
        cropnameList = list(df.groupby('pid')['cropname'].unique())

        # Unique crops in df:
        crops_unique = list(np.unique(cropnameList))
        
        # There may be multiple entries for the same pid on the same date for the same band (because of overlapping 
        # Sentinel tiles). Sort the dataframe to make sure that the entry with the highest pixel count is at the top for 
        # each set of duplicates, then remove all duplicates keeping only the top entry for each:
        df.sort_values(['pid','Date','band','count'], ascending=(True,True,True,False), inplace=True)
        df.drop_duplicates(['pid', 'Date', 'band'], inplace=True)

        # Remove any trailing spaces from band names:
        df['band'] = df['band'].str.strip()

        # Keep only the relevant columns including the column with the specified statistic to view:
        df = df[['pid','cropname','Date','band','count','orbit','satellite',stat]]

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(e)
        print('Error on line {}'.format(exc_tb.tb_lineno))
        sys.exit()  


        
    # Plot the time series:
    
    try:

        band_list = ['blue','green','nbr','nbr2','ndre','ndvi','nir','rel','relvi','re2','re3','re4','red','swir1',
                    'swir2','coh6_vh','coh6_vv','grd_vh','grd_vhonvv','grd_vv','grd_vv_vh','grd_vvonvh']

        # Determine which bands to plot:
        incorrect_bands = []
        if type(bands) == str:
            bands = list(bands.split(','))
            for band in bands:
                if band in band_list:
                    pass
                else:
                    incorrect_bands.append(band)

        if incorrect_bands:
            print("The following bands are not available: {}".format(incorrect_bands))
            sys.exit(1)               

        # Plot the data:
        for band in bands:
            dfBand = df[df['band'] == band]
            plt.figure(figsize=(15,5))
            # Iterate through each parcel and add a plot for it. 
            # Give it a unique colour based on its index position:
            for index, parcel in enumerate(dfBand['pid'].unique()):
                dfPid = dfBand[dfBand['pid'] == parcel]

                satellite = dfPid['satellite'].unique()      
                if satellite == 'S1':       
                    if orbit == 'A':
                        dfPid = dfPid[dfPid['orbit'] == 'A']
                        orbit_type = 'A'

                    elif orbit == 'D':
                        dfPid = dfPid[dfPid['orbit'] == 'D']
                        orbit_type = 'D'

                    else:
                        dfPid
                        orbit_type = 'A + D'

                if freq_avg is False:

                    # Apply moving average smoothing function:
                    if window_size:
                        rolling_mean = dfPid.groupby('pid')[stat].rolling(window=window_size).mean()
                        dfPid[stat] = rolling_mean.reset_index(level=0, drop=True)      
                        dfPid = dfPid.sort_values(['pid', 'Date'])     

                else:

                    # Aggregate the S1 data by frequency defined by user: Get the mean value for each week:
                    dfPid = dfPid.groupby([pd.Grouper(key='Date', freq = frequency), 'pid'])[stat].mean().reset_index(name = stat)

                crop = cropnameList[index][0]
                
                if class_colours is False:

                    if satellite == 'S1':
                        plt.plot(dfPid['Date'], dfPid[stat], linestyle = 'dashed', marker = 'o',
                                 color = plt.cm.tab20(index), label = (parcel, crop, orbit_type))
                    elif satellite == 'S2':
                        plt.plot(dfPid['Date'], dfPid[stat], linestyle = 'dashed', marker = 'o',
                                 color = plt.cm.tab20(index), label = (parcel, crop))

                else:

                    if satellite == 'S1':
                        plt.plot(dfPid['Date'], dfPid[stat], linestyle = 'dashed', marker = 'o',
                                 color = plt.cm.tab20(crops_unique.index(crop)+1), label = (parcel, crop, orbit_type))
                    elif satellite == 'S2':
                        plt.plot(dfPid['Date'], dfPid[stat], linestyle = 'dashed', marker = 'o',
                                 color = plt.cm.tab20(crops_unique.index(crop)+1), label = (parcel, crop))

                if logs == True:
                    # Run extractActivities function to return transect log for parcel:
                    dfActivities = db_queries.getLogActivities(pid=parcel, dataset=config.read(conf_file)[dataset])
                    singleActivity(dfActivities, ymin, ymax)

            if legend is True:
                plt.legend()

            if 'coh' in band:
                cardType = 'Coherence'
            elif 'grd' in band:
                cardType = 'Backscatter (Sigma 0)'
            else:
                cardType = 'Sentinel-2 Level-2A'

            plt.title(f"{cardType} time series")
            plt.xlabel('Date')
            plt.ylabel('{}\n{}'.format(band, stat))

            if set_ylim is True:
                plt.ylim((ymin,ymax))
                
            plt.show() 
    
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(e)
        print('Error on line {}'.format(exc_tb.tb_lineno))
        sys.exit() 



def plotTimeSeriesOutreach(df, stat, tstype, bands=None, start_date=None, end_date=None, weekly_avg=False,
                           window_size=None, separate_orbits=False):
    """
    This function creates time series graphs for the given parcel(s). Graphs will be plotted for any of the 
    following card types depending on the input dataframe: 's2', 'bs', 'c6'.
    The user can also determine whether to apply a rolling average to the data or a weekly average if desired.
    Additionally, the user can choose to plot DESC and ASC data separately or together.
    
    Parameters:
    
    - df: Dataframe - Dataframe with time series data for selected parcels
    - stat: String - The user can select from the following: 'mean', 'std', 'min', 'max',
            'p25', 'p50', 'p75'
    - tstype: String - Must be one of the following: 's2', 'bs', 'c6'
    - bands: String or List - Options are 'blue','green','nir','re1','re2','re3','re4','red','swir1','swir2',
            'NBR','NBR2','NDRE','NDVI','RE1VI' for s2, 'VV','VH','VV-VH','VVonVH','VHonVV' for bs, 'VV','VH' 
            for c6. 
            Default if none selected is 'NDVI' for s2 and 'VV' and 'VH' for bs/c6.
    - start_date: String - format must be '%Y-%m-%d'. Default if none specified is date of first entry
            in dataframe
    - end_date: String - format must be '%Y-%m-%d'. Default if none specified is date of last entry
            in dataframe
    - weekly_avg: Boolean - User can apply a weekly average to the data
    - window_size: Integer - User can apply a moving average smoothing function to the data with a 
            specified window size. Default is no moving average smoothing function
    - separate_orbits: Boolean - User can define whether to merge and plot the DESC and ASC orbits together or 
            whether to keep them separate and plot them separately.

    """

    # Input dataframe refinements:
    try:

        df = df.copy()

        # Convert the epoch timestamp to a datetime:
        df['date_part'] = df['date_part'].map(lambda e: datetime.fromtimestamp(e))

        # Get subset of dataframe if start_date and end_date specified:
        if start_date:
            df = df[df['date_part'] >= datetime.strptime(start_date, '%Y-%m-%d')]

        if end_date:
            df = df[df['date_part'] <= datetime.strptime(end_date, '%Y-%m-%d')]

        # Create an 'orbit' column with information about the Asc and Desc orbits:
        df['orbit'] = df['date_part'].apply(lambda s: 'D' if s.hour < 12 else 'A')


    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(e)
        print('Error on line {}'.format(exc_tb.tb_lineno))
        sys.exit()

        # Plot the time series:
    try:

        if weekly_avg is False:

            # ROLLING AVERAGE OR ORIGINAL TIME SERIES:

            for band in bands:
                dfBand = df[df['band'].str.contains(band)]
                plt.figure(figsize=(15, 5))
                # Iterate through each parcel and add a plot for it. 
                # Give it a unique colour based on its index position:
                for index, parcel in enumerate(dfBand['pid'].unique()):
                    dfPid = dfBand[dfBand['pid'] == parcel]

                    if separate_orbits is False:

                        # Apply moving average smoothing function:
                        if window_size:
                            rolling_mean = dfPid.groupby('pid')[stat].rolling(window=window_size).mean()
                            dfPid[stat] = rolling_mean.reset_index(level=0, drop=True)
                            dfPid = dfPid.sort_values(['pid', 'date_part'])
                        crop = str(np.unique(np.array(dfPid['crop'])))
                        plt.plot(dfPid['date_part'], dfPid[stat], linestyle='dashed', marker='o',
                                 color=plt.cm.tab20(index), label=(parcel, crop))

                    elif separate_orbits is True:

                        dfDesc = dfPid[dfPid['orbit'] == 'D']
                        dfAsc = dfPid[dfPid['orbit'] == 'A']

                        # Apply moving average smoothing function to DESC orbit:
                        if window_size:
                            rolling_mean = dfDesc.groupby('pid')[stat].rolling(window=window_size).mean()
                            dfDesc[stat] = rolling_mean.reset_index(level=0, drop=True)
                            dfDesc = dfDesc.sort_values(['pid', 'date_part'])
                        crop = str(np.unique(np.array(dfDesc['crop'])))
                        orbit = str(np.unique(np.array(dfDesc['orbit'])))
                        plt.plot(dfDesc['date_part'], dfDesc[stat], linestyle='dashed', marker='o',
                                 color=plt.cm.tab20(index), label=(parcel, crop, orbit))

                        # Apply moving average smoothing function to ASC orbit:
                        if window_size:
                            rolling_mean = dfAsc.groupby('pid')[stat].rolling(window=window_size).mean()
                            dfAsc[stat] = rolling_mean.reset_index(level=0, drop=True)
                            dfAsc = dfAsc.sort_values(['pid', 'date_part'])
                        crop = str(np.unique(np.array(dfAsc['crop'])))
                        orbit = str(np.unique(np.array(dfAsc['orbit'])))
                        plt.plot(dfAsc['date_part'], dfAsc[stat], linestyle='dashed', marker='o',
                                 color=plt.cm.tab20(index + 1), label=(parcel, crop, orbit))

            plt.legend()
            plt.title("{} time series".format(tstype))
            plt.xlabel('Date')
            plt.ylabel('{}\n{}'.format(band, stat))
            plt.show()

        else:

            # WEEKLY AVERAGES:

            # Group by pid and return the crop for each pid:
            pidGrBy = df.groupby('pid')
            pidCrops = pidGrBy['crop'].first()

            for band in bands:
                dfBand = df[df['band'].str.contains(band)]

                # Reset the row index:
                dfBand = dfBand.reset_index()

                plt.figure(figsize=(15, 5))

                if separate_orbits is False:

                    # Aggregate the S1 data by week: Get the mean value for each week:
                    dfWeeks = dfBand.groupby([pd.Grouper(key='date_part', freq='W'), 'pid'])[stat].mean().reset_index(
                        name='mean')

                    # Join on the crop for each pid:
                    dfWeeks = pd.merge(dfWeeks, pidCrops, on='pid')

                    for index, parcel in enumerate(dfWeeks['pid'].unique()):
                        dfPid = dfWeeks[dfWeeks['pid'] == parcel]

                        crop = str(np.unique(np.array(dfPid['crop'])))

                        plt.plot(dfPid['date_part'], dfPid[stat], linestyle='dashed', marker='o',
                                 color=plt.cm.tab20(index), label=(parcel, crop))

                elif separate_orbits is True:

                    # Separate the dataframe into Descending and Ascending rows:
                    dfDesc = dfBand[dfBand['orbit'] == 'D']
                    dfAsc = dfBand[dfBand['orbit'] == 'A']

                    # Aggregate the S1 data by week: Get the mean value for each week:
                    dfWeeksDesc = dfDesc.groupby([pd.Grouper(key='date_part', freq='W'), 'pid'])[
                        stat].mean().reset_index(name='mean')
                    dfWeeksAsc = dfAsc.groupby([pd.Grouper(key='date_part', freq='W'), 'pid'])[stat].mean().reset_index(
                        name='mean')

                    # Join on the crop for each pid:
                    dfWeeksDesc = pd.merge(dfWeeksDesc, pidCrops, on='pid')
                    dfWeeksAsc = pd.merge(dfWeeksAsc, pidCrops, on='pid')

                    for index, parcel in enumerate(dfBand['pid'].unique()):
                        dfPidDesc = dfWeeksDesc[dfWeeksDesc['pid'] == parcel]
                        dfPidAsc = dfWeeksAsc[dfWeeksAsc['pid'] == parcel]

                        crop = str(np.unique(np.array(dfBand['crop'])))

                        orbitDesc = 'D'
                        plt.plot(dfPidDesc['date_part'], dfPidDesc[stat], linestyle='dashed', marker='o',
                                 color=plt.cm.tab20(index), label=(parcel, crop, orbitDesc))

                        orbitAsc = 'A'
                        plt.plot(dfPidAsc['date_part'], dfPidAsc[stat], linestyle='dashed', marker='o',
                                 color=plt.cm.tab20(index + 1), label=(parcel, crop, orbitAsc))

                        # plt.legend()
            plt.title("{} time series".format(tstype))
            plt.xlabel('Date')
            plt.ylabel('{}\n{}'.format(band, stat))
            plt.show()

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(e)
        print('Error on line {}'.format(exc_tb.tb_lineno))
        sys.exit()


def sampleParcelsList(in_shp, crop, crop_claim_col, subs_col, id_col, max_parcels, seed):
    """
    This function returns a sample list of Outreach parcel IDs for a given input crop type with 
    no subdivisions included
    
    Parameters:
    
    - in_shp: String - filepath to input shapefile
    - crop: String - Crop name as it appears in CropClaime column
    - crop_claim_col: String - Name of crop claim column in the shapefile
    - subs_col: String - Name of subdivisions column in the shapefile
    - id_col: String - Name of ID column in the shapefile
    - max_parcels: Integer - The maximum number of parcels to return for the given input crop type
    - seed: Integer - Seed number to set for repeatability
    
    """

    # Read in shapefile:
    shp = fiona.open(in_shp)

    # Create a list with every parcel ID for the given input crop:
    pidList = []
    for feature in shp:
        if feature['properties'][crop_claim_col] == crop and feature['properties'][subs_col] == 'N':
            pidList.append(feature['properties'][id_col])

    # Select a random sample of parcels up to the max_parcels number set by the user:
    random.seed(seed)
    if len(pidList) >= max_parcels:
        pidList = sample(pidList, max_parcels)

    return pidList


def singleActivity(dfActivities, ymin=0, ymax=1):

    ymid = (ymax + ymin)/2

    # List of possible events:
    event_dict = {'harvested crop':'red', 'baled straw':'green', 'un baled straw':'orange', 'stubble present':'blue', 
                  'cultivated stubble':'yellow'}
    #dates = []    

    for i in dfActivities.items():
        for key, value in i[1].items():
            new_value = str(value).split('/')[0]
            i[1][key] = new_value

    activities_dict = {}
    index = 0
    for column, row in dfActivities.items():
        for activity in row.values():
            for event in event_dict.keys():
                if event == str(activity).lower():
                    index = index+1
                    activities_dict[index] = {event: column}
               
    if len(activities_dict) > 0:
        for nested_dict in activities_dict.items():
            activity = nested_dict[1]
            for key, value in activity.items():
                plt.axvline(datetime.strptime(value, '%Y%m%d'), 0, 1, color = event_dict[key],
                            linestyle='dashed')
                plt.text(datetime.strptime(value, '%Y%m%d'), ymid, key, color = event_dict[key],
                         rotation=270)
    
    ## Extract event dates:
    #for column, row in dfActivities.items():
    #    for activity in row.values():
    #        for event in events:
    #            if event in str(activity).lower():
    #                dates.append(column)
    #if len(dates) > 0:
    #    event_date = dates[0]
    #    plt.axvline(datetime.strptime(event_date, '%Y%m%d'), 0, 1, color = 'red')
