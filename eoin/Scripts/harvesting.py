import sys
import os
import time
import pandas as pd
import numpy as np
import fiona
import matplotlib.pyplot as plt
from random import sample
import random
from datetime import datetime
from datetime import timedelta
import more_itertools as mit

import time_series_local as ts

from database.db import tables, execute_query, config
from database import db_queries
from database.utils import keyword_search

pd.options.mode.chained_assignment = None  # default='warn'
kywd = keyword_search(os.path.join(os.getcwd(), 'config/config_file.txt'), ['DS_config'])
path_conf = os.path.dirname(kywd['DS_config'])
conf = os.path.basename(kywd['DS_config'])
conf_file = os.path.join(os.getcwd(), path_conf, conf)


def thresholdingAlgo(y, lag, threshold, influence):
    
    """Reference:
    Brakel, J.P.G. van (2014). "Robust peak detection algorithm using z-scores". 
    Stack Overflow. Available at: 
    https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data/22640362#22640362 
    (version: 2020-11-08).
    
    The original function referenced above has been modified to suit the purposes of AMS harvesting detection analysis.
    The function carries out the following key operations:
    
    - A mean value based on the previous n number of dates (determined by the lag parameter set by the user) is 
      calculated for each date and added to an average filter array.
    - The std dev values based on the previous n number of dates is calculated for each date and added to a std
      dev filter array.
    - Each time series value is compared in turn to its corresponding average filter value. If the time series value
      minus the average filter value is greater than the corresponding std dev value multiplied by the threshold
      set by the user (number of standard deviations) and the time series value itself is greater than the average 
      value, then a signal with a value of 1 is produced. This indicates a peak. Signals with values of -1 or 0 
      indicate no peak.
    - For signals with a value of 1, a corresponding 'signalStrength' value is also produced. This value is 
      calculated as the difference between the given time series value minus the corresponding (average filter value
      plus (the threshold multiplied by the std deviation value)).
    - The output from the function is a dictionary with the signals, signalStrenghts, avgFilter and stdFilter.
    
     Parameters:
    
    - y: Array - DF Time series 'stat' column converted to an array
    - lag: Integer - The number of previous dates to calculate the smoothed average filter from for each point
    - threshold: Integer/Float - The number of standard deviations from the moving mean above which the algorithm 
      will classify a new datapoint as being a signal
    - influence: Integer/Float - determines the influence of signals on the algorithm's detection threshold. If 
      put at 0, signals have no influence on the threshold, such that future signals are detected based on a 
      threshold that is calculated with a mean and standard deviation that is not influenced by past signals.
      Value must be between 0 and 1. 
    """
    
    try:
        # Create an array of zeros for the signals (signals indicate whether a datapoint is x stanadard deviations above 
        # the moving mean and threshold set or not. Each datapoint will get a signal value of either 1, 0 or -1):
        signals = np.zeros(len(y))
        # Create an array of zeros for signalStrength. This will be used to hold a value for the strength of the signal 
        # above the thresholded mean:
        signalStrength = np.zeros(len(y))
        # Create an array from the input time series:
        filteredY = np.array(y)
        # Create a list of zeros for what will become an average filter:
        avgFilter = [0]*len(y)
        # Create a list of zeros for what will become a std dev filter:
        stdFilter = [0]*len(y)
        # Calculate the first value for the average filter. This will be in the position which is equal to the lag set
        # by the user (note that it is specified here as [lag - 1] because lists and arrays use different indexing - the
        # former starts at 0, the latter at 1). The value produced in this position of the average filter will be the mean
        # of the all the numbers in the time series data up to the lag set by the user:
        avgFilter[lag - 1] = np.mean(y[0:lag])
        # As above, calculate the first value for the std dev filter:
        stdFilter[lag - 1] = np.std(y[0:lag])
        # Loop through all remaining values in the time series and for each one, check whether the time series value minus
        # the corresponding average filter value is greater than the corresponding std dev value multiplied by the threshold
        # set by the user. If it isn't then give the signal for this datapoint a value of 0. If it is, then check whether 
        # the time series datapoint is itself greater than the corresponding average filter value. If it is, give the signal 
        # a value of 1, otherwise give it a value of -1. Finally, the next value for the average and std dev filters is then
        # calculated again based on the number of previous time series points specified in the lag, and also depending on 
        # the influence parameter set by the user:
        for i in range(lag, len(y)):
            if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
                if y[i] > avgFilter[i-1]: #and y[i] >= 0.4:
                    signals[i] = 1
                    #signalStrength[i] = round((y[i] - avgFilter[i-1])/stdFilter[i-1],2) # This calcualtes number of std deviations above the mean
                    signalStrength[i] = round(y[i] - (avgFilter[i-1] + (threshold * stdFilter[i-1])),4) # This calculates actual difference between time series datapoint and the threshold set by user
                else:
                    signals[i] = -1

                filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
                avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
                stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])
            else:
                signals[i] = 0
                filteredY[i] = y[i]
                avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
                stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])

        return dict(signals = np.asarray(signals),
                    signalStrength = np.asarray(signalStrength),
                    avgFilter = np.asarray(avgFilter),
                    stdFilter = np.asarray(stdFilter))
    
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(e)
        print('Error on line {}'.format(exc_tb.tb_lineno))
        sys.exit()
        

def signalPlots(dataset, y, dates, result, crop, pid, threshold, config_file, orbit, band, stat, 
               logs=True, set_ylim=False, ymin=None, ymax=None):
    
    """
    This function plots the results from the thresholdingAlgo() function. Specifically, it creates two plots:
    1) The first plot shows the original time series in dark blue, the average filter in light blue, and the 
        average filter + the threshold * std filter and the average filter - the threshold * std filter both in 
        green. Additionally, the GC events are plotted with a red vertical line.
    2) The second plot shows the signals returned.
    
    Parameters:
    
    - dataset: String - Name of the dataset as defined in datasets.json config file
    - y: Array - DF Time series 'stat' column converted to an array
    - dates: Array - DF Time series 'date_part' column converted to an array
    - result: Dictionary - Dictionary result returned from the thresholdingAlgo() function
    - crop: String - Crop type for the given parcel
    - pid: List - Parcel ID
    - threshold: Integer/Float - The number of standard deviations from the moving mean above which the algorithm 
      will classify a new datapoint as being a signal
    - config_file: String - Path to config file
    - orbit: String - Options are 'A', 'D' or 'D & A'
    - band: String - Options are 'VV','VH'
    - stat: String - The user can select from the following: 'mean', 'std', 'min', 'max',
            'p25', 'p50', 'p75'
    - logs: Boolean - True or False: The user can also decide to plot log activities if they exist.
            Default is False
    - set_ylim: Boolean - User can decide to set ymin and ymax values for the y axis.
            Default is False
    - ymin: Float - Minimum value for Y axis if required.
            Default is None
    - ymax: Float - Maximum value for Y axis if required
            Default is None
   
    """
    
    try:
        
        plt.figure(figsize=(15,8))

        # First plot:
        plt.subplot(211)

        # Plot original time series
        plt.plot(dates, y, label = (pid, crop, orbit))

        # Plot average filter
        plt.plot(dates, result["avgFilter"], color="cyan", lw=2)

        # Plot average filter + threshold * std dev filter:
        plt.plot(dates, result["avgFilter"] + (threshold * result["stdFilter"]), color="green", lw=2)

        # Plot average filter - threshold * std dev filter:
        plt.plot(dates, result["avgFilter"] - (threshold * result["stdFilter"]), color="green", lw=2)

        #plt.ylim(round(y.min() -0.1, 2),round(y.max() +0.1, 2))

        # Add GC events:
        if logs is True:
            # Run extractActivities function to return transect log for parcel:         
            dfActivities = db_queries.getLogActivities(pid=pid, dataset=config.read(conf_file)[dataset])
            ts.singleActivity(dfActivities, ymin, ymax)



        plt.legend()
        plt.ylabel('{}\n{}'.format(band, stat))
        if set_ylim is True:
            plt.ylim((ymin,ymax))

        # Second plot: Signals
        plt.subplot(212)
        plt.step(dates, result["signals"], color="red", lw=2)
        plt.ylim(-1.5, 1.5)

        # Display plots:
        plt.show()        

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(e)
        print('Error on line {}'.format(exc_tb.tb_lineno))
        sys.exit()


def harvestingEvents(result_df, stat, min_coh):
    
    """
    This function produces a number of metrics related to each of the possible harvesting events detected in the 
    thresholdingAlgo() function. The function carries out the following key procedures:
    
    1) Based on the signalStrengths returned from the thresholdingAlgo() function, a list is created with 
        consecutive index numbers grouped into sublists. Each sublist represents a possible harvesting event.
    2) Any sublists with less than two elements are removed as these are considered non-events.
    3) A list with sublist groups of signalStrengths is created with each sublist representing a possible
        harvesting event. A dictionary is then created of events and their corresponding signalStrengths for each
        date within the given event.
    4) Next, any events which don't have at least one value above the minimum coherence value set by the user
        are removed.
    5) Next, three metrics are produced to provide information on each event to help determine the nature and 
        significance of each event: a) the area of each event above the thresholded mean is calculated; 
        b) the max value above the thresholded mean is calculated; c) the percentage of values above the mininum 
        coherence value set by the user is calculated. These are all stored in a dictionary (eventDict).
    6) Finally, the original dates are extracted for each event and stored in a dictionary.
    7) There are four dictionary outputs from the function: 
        - sigGroupDict (dictionary with signalStrengths for each event) 
        - eventDict (dictionary with three metrics for assessing quality and character of each event) 
        - cohGroupValues (dictionary with original coherence values for each event) 
        - datesGroupDict (dictionary with original dates for each event)
        
    Parameters:
    
    - mowing_result_df: DataFrame - dataframe including original time series values plus outputs from 
        thresholdingAlgo() function.
    - stat: String - The user can select from the following: 'mean', 'std', 'min', 'max',
            'p25', 'p50', 'p75'
    - min_coh: Float - This refers to the minimum original coherence value below which the user loses
        significant confidence in the event being a mowing event.
    
    """
    
    try:
    
        # Get the signalStrength returned from the mowing_detection() function and convert it to a list:
        sigStrength = list(result_df['signalStrength'])
        # Remove all elements from the list with a value of zero (i.e. 'non events') and return the index values
        # of the remaining elements:
        removeZeros = [i for i, e in enumerate(sigStrength) if e != 0]
        # Group the index values by consecutive numbers using the consecutive_groups() function. This function groups
        # numbers if they are consecutive to one another. Each group is returned as a separate list:
        sigGroups = [list(group) for group in mit.consecutive_groups(removeZeros)]
        # Remove any list groups with less than 2 elements as these are highly unlikely to be mowing events.
        # First, create a list with the index numbers to remove.
        # Second, apply this to the sigGroups:
        remove_indices = [i for i, e in enumerate(sigGroups) if len(e) <= 2]
        sigGroupsMain = [e for i, e in enumerate(sigGroups) if i not in remove_indices]


        # Use the sigGroupsMain list to create a dictionary with the groups of signalStrength values:
        sigGroupDict = {}
        for i, x in enumerate(sigGroupsMain):
            sigValues = []
            for index in x:
                sigValues.append(sigStrength[index])
            sigGroupDict[i] = sigValues


        # Extract the original coherence values for the event groups and add two additional values at the end 
        # of each group (this is because the thresholded average filter tends to cut off many of the values on the
        # downward slope of an event):
        cohGroupsMain = [e for i, e in enumerate(sigGroups) if i not in remove_indices]
        for i, l in enumerate(cohGroupsMain):
            cohGroupsMain[i].append(cohGroupsMain[i][-1]+1)
            cohGroupsMain[i].append(cohGroupsMain[i][-1]+1)           

        cohGroupValues = {}
        for i, x in enumerate(cohGroupsMain):
            cohValues = []
            for index in x:
                if index < len(result_df):
                    cohValues.append(list(result_df[stat])[index])
                cohGroupValues[i] = cohValues               

        # Remove any events which don't have at least one value which is greater than the min_coh value defined 
        # by the user:
        insignificant_event_list = []
        for key, value in cohGroupValues.items():
            min_coh_list = []
            for x in value:
                if x > min_coh:
                    min_coh_list.append(1)
                else:
                    min_coh_list.append(0)
            if max(min_coh_list) == 0:
                insignificant_event_list.append(key)

        for x in insignificant_event_list:
            cohGroupValues.pop(x, None)
            sigGroupDict.pop(x, None)
                

        # Get the area of each signal event above the thresholded average filter; 
        # Get the max value above the thresholded average filter for each signal group;
        # Get the count of values above a value set by the user for each signal group.
        # Include all of these in a dictionary:
        eventDict = {}
        for key in sorted(sigGroupDict.keys()):
            event = sigGroupDict.get(key) # return each set of signals as an 'event' list
            event.insert(0,0) # Add a zero at the first index
            event.insert(len(event),0) # Add a zero at the last index
            area = round(np.trapz(event, dx=1),2) # Get the area of the event using numpy trapz function
            eventDict[key] = {}
            eventDict[key]['area'] = area
            eventDict[key]['max_above_threshold'] = max(event)
        
        # Calculate the percentage of coherence values in an event which are above the min_coh value:
        highCount = []
        lowCount = []
        for key, value in sorted(cohGroupValues.items()):
            for v in value:
                if v >= min_coh:
                    highCount.append(v)
                else:
                    lowCount.append(v)
            eventDict[key]['highCount'] = int(round((len(highCount)/(len(highCount)+len(lowCount)))*100,0))            

        # Extract the original dates for the event groups:
        datesGroupsMain = [e for i, e in enumerate(sigGroups) if i not in remove_indices] 

        datesGroupDict = {}
        for i, x in enumerate(cohGroupsMain):
            datesValues = []
            for index in x:
                if index < len(result_df):
                    datesValues.append(list(result_df['Date'])[index])
                datesGroupDict[i] = datesValues

        # Add the start and end dates of each detected event to the eventDict:
        for key in eventDict.keys():
            eventDict[key]['startDate'] = datetime.strftime(datesGroupDict[key][0], '%Y-%m-%d')
            eventDict[key]['endDate'] = datetime.strftime(datesGroupDict[key][-1], '%Y-%m-%d') 

        return sigGroupDict, eventDict, cohGroupValues, datesGroupDict

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(e)
        print('Error on line {}'.format(exc_tb.tb_lineno))
        sys.exit()



def harvestDetection(dataset, df, band, start_date, end_date, stat, window_size=None, orbit=None, lag=10, threshold=3, 
                    influence=0.1, show_signals=False, show_signal_plots=False, config_file=conf_file, 
                    min_coh=0.4, logs=False, set_ylim=False, ymin=None, ymax=None):

    """
    This function is the primary function for detecting harvesting events. The function carries out the following key procedures:
    
    1) The input dataframe with the extracted time series data for a parcel is further processed to filter it where set by
        the user by orbit, bands, start and end date.
    2) The time series is smoothed if set by the user
    3) The thresholdingAlgo() function is run to return the signals and signal strengths for the given time series
    4) The signals are displayed and signal plots are plotted if set by the user
    5) The harvestingEvents() function is run to convert the signals into identifiable events with a series of metrics
        associated to each event.
        
    Parameters:

    - dataset: String - the table identifier in postgres to read in the relevant data from. This dataset must be referenced
        in the appropriate datasets.json config file
    - df: Dataframe - Dataframe with extracted parcel time series from the parcelTimeSeriesDfLocal() function in the
        time_series.py script
    - band: String - Options are 'VV','VH'
    - start_date: String - format must be '%Y%m%d'. Default if none specified is date of first entry
        in dataframe. ***Note that this date should be set prior to the actual date the user wishes to monitor from as
        the thresholdingAlgo() function applies a lag. Depending on the size of the lag, the effective monitoring start date
        will be the date of the first recorded coherence value after the lag. For example, if the start_date is set as
        20200501 and a lag of 10 is set, then the effective monitoring date will be 10 captured dates after the start_date.
        If data has been captured approx every 3 days, then the effective monitoring date will be 20200531
    - end_date: String - format must be '%Y%m%d'. Default if none specified is date of last entry
        in dataframe
    - stat: String - The user must choose a single stat but can select from any of the following: 'min', 'max', 'mean',
        'median', 'std', 'p5', 'p25', 'p75', 'p95'
    - window_size: Integer -  User can apply a moving average smoothing function to the data with a 
        specified window size. Default is None
    - orbit: String - User can set whether to analyse just one specific orbit. Options are 'D' (descending), 'A' (ascending)
        or None (both). Default is None
    - lag: Integer - The number of previous dates to calculate the smoothed average filter from for each point
    - threshold: Integer/Float - The number of standard deviations from the moving mean above which the algorithm 
        will classify a new datapoint as being a signal
    - influence: Integer/Float - determines the influence of signals on the algorithm's detection threshold. If 
        put at 0, signals have no influence on the threshold, such that future signals are detected based on a 
        threshold that is calculated with a mean and standard deviation that is not influenced by past signals.
        Value must be between 0 and 1.
    - show_signals: Boolean - The User can set whether to show the signal result array for a parcel.
        Default is False
    - show_signal_plots: Boolean - The user can set whether to display the signal plots which show the results of the
        thresholdingAlgo() function. Default is False
    - config_file: String - filepath of the config_file (datasets.json). Default is conf_file as defined in this script
    - min_coh: Float - The minimum coherence value below which no returned signal will be considered as a possible
        harvesting event. Default is 0.4 (for VV median)
    - logs: Boolean - The user can select to display transect data information where available. Default is False
    - set_ylim: Boolean - The user can select whether to set ymin and ymax values for the signal plots if desired.
        Default is False
    - ymin: Float - If set_ylim is True, the user can define the minimum value to be displayed on the Y axis.
        Default is None
    - ymax: Float - If set_ylim is True, the user can define the maximum value to be displayed on the Y axis.
        Default is None    

    """
    
    try:

        df = df.copy()

        # Determine the orbit direction (Descending or Ascending) for S1:
        df['satellite'] = df['systemindex'].apply(lambda s: 'S1' if s[:2] == 'S1' else 'S2')

        df_index = df.index

        dfs1_index = df_index[df['satellite'] == 'S1']

        # Create an 'orbit' column with orbit information for each date:
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

        # Select specific orbit if requested:
        if orbit == 'A':
            df = df[df['orbit'] == 'A']
        elif orbit == 'D':
            df = df[df['orbit'] == 'D']

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

        # Apply a rolling mean to the time series if specified:
        if window_size:
            rolling_mean = df.groupby('pid')[stat].rolling(window=window_size).mean()
            df[stat] = rolling_mean.reset_index(level=0, drop=True)      
            df = df.sort_values(['pid', 'Date'])

        # Remove any rows with NaNs (there will be rows with NaNs if a rolling mean has been applied):
        df = df.dropna()
        
        pid = int(df['pid'].unique())

        # Convert the stat and date_part columns into arrays:
        ts_array = np.array(df[stat])
        dates = np.array(df['Date'])

        # Run the thresholdingAlgo() function to return the signals and signal strengths of the time series:
        harvest_result = thresholdingAlgo(y=ts_array, lag=lag, threshold=threshold, influence=influence)

        # Show the signal results if required:
        if show_signals is True:
            print(harvest_result['signals'])
            print(harvest_result['signalStrength'])

        # Show the signal plots if required:
        if show_signal_plots is True:
            signalPlots(dataset=dataset, y=ts_array, dates=dates, result=harvest_result, crop=crops_unique, 
                        pid=pid, threshold=threshold, config_file=conf_file, orbit=orbit, 
                        band=band, stat=stat, logs=logs, set_ylim=set_ylim, ymin=ymin, ymax=ymax)
            
        # add the results (signals, signalStrengths, avgFilter and stdFilter) from the thresholdingAlgo() function 
        # as columns in the dataframe. Join by index values:
        df = df.reset_index(drop=True)
        result_df = df.merge(pd.DataFrame.from_dict(harvest_result), left_index=True, right_index=True)

        # Convert the signals into identifiable events and return metrics related to these events:
        sigGroupDict, eventDict, cohGroupValues, datesGroupDict = harvestingEvents(result_df, 
                                                                               stat, min_coh)
            
        return sigGroupDict, eventDict, cohGroupValues, datesGroupDict
        
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(e)
        print('Error on line {}'.format(exc_tb.tb_lineno))
        sys.exit()



def computeConfidence(event_dict, coh_groups, date_groups):
    
    """
    ***NOTE THAT THIS FUNCTION HAS BEEN BUILT TO PROCESS MEDIAN VALUES FOR VV COHERENCE ONLY. OTHER STATS AND BANDS
    WOULD REQUIRE AMENDMENTS***
    
    This functions creates a confidence score for each event based primarily on the metrics computed in the 
    harvestingEvents() function. A range of confidence scores are produced for each metric and finally an overall
    confidence to classify an event as a harvesting event is derived.
    The metrics taken into consideration are:
    1) Event 'area'; 2) Event 'max above threshold' (the highest value above the threshold value);
    3) Event 'high count': The percentage of values above the minimum coherence value set by the user
    
    Parameters:
    
    - event_dict: Dictionary - eventDict dicitonary with event metrics output from the mowingEvents() function.
    - coh_groups: Dictionary - cohGroupValues dictionary output from the mowingEvents() function with original 
        coherence values for each event.
    - date_groups: Dictionary - datesGroupDict dictionary output from the mowingEvents() function with original
        dates for each event.
    
    """
    
    try:
    
        if event_dict:

            # Assign confidence values based on area, max, highCount  and min_coh values:
            for key in event_dict.keys():

                if event_dict[key]['area'] < 0.295:
                    area_conf = 0.2
                elif event_dict[key]['area'] >= 0.295 and event_dict[key]['area'] < 0.59:
                    area_conf = 0.4
                elif event_dict[key]['area'] >= 0.59 and event_dict[key]['area'] < 0.885:
                    area_conf = 0.6
                elif event_dict[key]['area'] >= 0.885 and event_dict[key]['area'] < 1.18:
                    area_conf = 0.8
                elif event_dict[key]['area'] >= 1.18:
                    area_conf = 1

                if event_dict[key]['max_above_threshold'] < 0.0525:
                    max_conf = 0.2
                elif event_dict[key]['max_above_threshold'] >= 0.0525 and event_dict[key]['max_above_threshold'] < 0.105:
                    max_conf = 0.4
                elif event_dict[key]['max_above_threshold'] >= 0.105 and event_dict[key]['max_above_threshold'] < 0.1575:
                    max_conf = 0.6
                elif event_dict[key]['max_above_threshold'] >= 0.1575 and event_dict[key]['max_above_threshold'] < 0.21:
                    max_conf = 0.8
                elif event_dict[key]['max_above_threshold'] >= 0.21:
                    max_conf = 1

                if event_dict[key]['highCount'] < 18.75:
                    highCount_conf = 0.2
                elif event_dict[key]['highCount'] >= 18.75 and event_dict[key]['highCount'] < 37.5:
                    highCount_conf = 0.4
                elif event_dict[key]['highCount'] >= 37.5 and event_dict[key]['highCount'] < 54.45:
                    highCount_conf = 0.6
                elif event_dict[key]['highCount'] >= 54.45 and event_dict[key]['highCount'] < 75:
                    highCount_conf = 0.8
                elif event_dict[key]['highCount'] >= 75:
                    highCount_conf = 1

                # Calculate the overall confidence that each event is a mowing event:
                overall_conf = round((area_conf + max_conf + highCount_conf)/3, 2)

                event_dict[key]['conf'] = overall_conf
                
                complete_event_dict = dict(event_dict)
                if complete_event_dict[key]['conf'] == 0:
                    del complete_event_dict[key]

            return complete_event_dict
    
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(e)
        print('Error on line {}'.format(exc_tb.tb_lineno))
        sys.exit()
