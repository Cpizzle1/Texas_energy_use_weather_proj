import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

def city_weather_clean(df):
    df['Cloud_numerical'] =  df['cloud']
    d1 =  {
    'Fair':0
    ,'Mostly Cloudy':2
    ,'Cloudy':1
    ,'Partly Cloudy':1
    ,'Light Rain':2
    , 'Light Drizzle':2
    ,'Rain':2
    ,'Light Rain with Thunder':2
    ,'Heavy T-Storm':2
    ,'Thunder':2
    , 'Heavy Rain':2
    ,'T-Storm':2
    , 'Fog':2
    , 'Mostly Cloudy / Windy':2
    , 'Cloudy / Windy':2
    , 'Haze':1
    , 'Fair / Windy':0
    , 'Partly Cloudy / Windy':1
    , 'Light Rain / Windy':2
    , 'Heavy T-Storm / Windy':2
    , 'Heavy Rain / Windy':2
    , 'Widespread Dust':1
    , 'Thunder and Hail':2
    ,'Thunder / Windy':2
    ,'Blowing Dust':1
    , 'Patches of Fog':1
    , 'Blowing Dust / Windy':1
    , 'Rain / Windy':2
    , 'Fog / Windy':2
    , 'Light Drizzle / Windy':2
    , 'Haze / Windy':1
    ,'Light Snow / Windy':1
    , 'Light Snow':1
    ,'T-Storm / Windy':2
    ,'Light Sleet':1
}
    df['Cloud_numerical'].replace(d1, inplace= True)
    
    df['new_hour_date'] = df['hour'] + ' '+  df['Date']
    df['New_datetime'] = pd.to_datetime(df['new_hour_date'],infer_datetime_format=True, format ='%m/%d/%Y %H')
    df['time_rounded'] = df['New_datetime'].dt.round('H').dt.hour
    df['time_rounded'] = df['time_rounded'].apply(str)
    df['time_rounded2'] = df['Date'] + ' '+  df['time_rounded']
    df['time_rounded4']= df['time_rounded2'].apply(lambda x:f'{x}:00:00')
    df['New_datetime2'] = pd.to_datetime(df['time_rounded4'],infer_datetime_format=True,format ='%m/%d/%Y %H')
    df['New_datetime'] = pd.to_datetime(df['New_datetime'],infer_datetime_format=True,format ='%m/%d/%Y %H')
    

    
    pd_series_precip = df['precip']
    precip_lst = []
    for string in  pd_series_precip:
        string = string.replace(u'\xa0in','')
        precip_lst.append(string)
    results_precip = pd.Series(precip_lst)
    df['precip1']= results_precip
    df['precip1'] = df['precip1'].astype(float)
    
    pd_series_dew = df['dew']
    dew_lst = []
    for string in pd_series_dew:
        string = string.replace(u'\xa0F','')
        
        dew_lst.append(string)
    results = pd.Series(dew_lst)
    df['dew1']= results
    df['dew1'] = df['dew1'].astype(float)
    
    pd_series_wind = df['wind_speed']
    wind_lst = []
    for string in pd_series_wind:
        string = string.replace(u'\xa0mph','')
        if string == '0.00\xa0':
            string = '0.00'
        wind_lst.append(string)
    
    results = pd.Series(wind_lst)
    df['wind1']= results
    df['wind1'] = df['wind1'].astype(float)
    
    pd_series_temp = df['temp']
    temp_lst = []
    for string in pd_series_temp:
        string = string.replace(u'\xa0F','')
        temp_lst.append(string)
    results = pd.Series(temp_lst)
    df['temp1']= results
    df['temp1'] = df['temp1'].astype(float)
    
    pd_series_pressure =df['pressure']
    pressure_lst = []
    for string in pd_series_pressure:
        string = string.replace(u'\xa0in','')
        if string == '0.00\xa0':
            string = '0.00'
        pressure_lst.append(string)
    
    results = pd.Series(pressure_lst)
    df['pressure1']= results
    df['pressure1'] = df['pressure1'].astype(float)
    
    pd_series_humidity = df['humidity']
    humidity_lst = []
    for string in pd_series_humidity:
        string = string.replace(u'\xa0%','')
        humidity_lst.append(string)
    
    results = pd.Series(humidity_lst)
    df['humdity1']= results
    df['humdity1'] = df['humdity1'].astype(float)
    
    
    
    del df['hour']
    del df['Date']
    del df['new_hour_date']
    del df['New_datetime']
    del df['time_rounded']
    del df['time_rounded2']
    del df['time_rounded4']
    del df['temp']
    del df['dew']
    del df['humidity']
    del df['wind_speed']
    del df['pressure']
    del df['precip']
    del df['cloud']
    
    return df


if __name__ == '__main__':

    
    dallas = pd.read_csv('~/Desktop/Dallas_YEAR_SCRAPE.csv')
    dallas_clean =  city_weather_clean(dallas)
    print(dallas_clean)

