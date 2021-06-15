import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import scipy.stats as stats
from pandas.plotting import scatter_matrix
import statsmodels.api as sm
import datetime as dt


def change_cols_to_floats(dataframe,lst):
    
    for i in lst:
        dataframe[i] = dataframe[i].str.replace(',', '')
        dataframe[i] = dataframe[i].astype(float)
    return dataframe
def make_date_time_col(df):
    df['Hour Number'] = df_total['Hour Number'].replace(24, 0)
    df['Hour Number'] = df_total['Hour Number'].replace(25, 0)
    df['Data Date']= df['Data Date'].astype(str)
    df['Hour Number'] = df['Hour Number'].astype(str)
    df['New_datetime'] = df['Data Date'].map(str) + " " + df['Hour Number']
    df['Hour Number'] = df['Hour Number'].astype(int)
    
    return df

def make_hourly_demand_means(df,lst):
    d = {}
    for i in lst:
        filt =df['Hour Number']==i
        d[i] = df.loc[filt]['Demand (MW)'].mean()
    return d

def graph_maker_for_energy_type_by_hour(df,column, lst = np.arange(0,24)):
    
    d= {}
    for i in lst:
        filt =df['Hour Number']==i
        hour_avg = df.loc[filt][column].mean()
        d[i]=hour_avg
    x = d.keys()
    y = d.values()
    fig, ax =plt.subplots(figsize = (8,8))
    ax.plot(x, y)
    ax.set_title(column)
    ax.set_xlabel('Time (Hours in Day)')
    ax.set_xticks(lst)
    
    
    plt.show()

def temp_column_convert_float2(pd_series):
    lst = []
    for string in pd_series:
        string = string.replace(u'\xa0F','')
        
        lst.append(string)
    
    results = pd.Series(lst)
    return results 

def press_column_convert_float2(pd_series):
    lst = []
    for string in pd_series:
        string = string.replace(u'\xa0in','')
        if string == '0.00\xa0':
            string = '0.00'
        lst.append(string)
    
    results = pd.Series(lst)
#     results2 = results.astype(float)
    return results

def wind_column_convert(pd_series):
    lst = []
    for string in pd_series:
        string = string.replace(u'\xa0mph','')
        if string == '0.00\xa0':
            string = '0.00'
        lst.append(string)
    
    results = pd.Series(lst)
#     results2 = results.astype(float)
    return results

def dew_column_convert_float2(pd_series):
    lst = []
    for string in pd_series:
        string = string.replace(u'\xa0F','')
        
        lst.append(string)
    
    results = pd.Series(lst)
    return results 
def change_cols_to_floats2(dataframes_lst):
    floated_dfs =[]
    for df in dataframes_lst:
        lst_cols = ['Demand (MW)','Net Generation (MW) from Natural Gas', 'Net Generation (MW) from Nuclear',
                    'Net Generation (MW) from All Petroleum Products',
                    'Net Generation (MW) from Hydropower and Pumped Storage', 
                    'Net Generation (MW) from Solar', 'Net Generation (MW) from Wind', 
                    'Net Generation (MW) from Other Fuel Sources','Net Generation (MW)',
                    'Demand Forecast (MW)', 'Total Interchange (MW)', 
                    'Net Generation (MW) (Adjusted)','Net Generation (MW) from Coal',
                    'Sum(Valid DIBAs) (MW)','Demand (MW) (Imputed)',
                    'Net Generation (MW) (Imputed)','Demand (MW) (Adjusted)']
        for i in lst_cols:
            df[i] = df[i].str.replace(',', '')
            df[i] = df[i].astype(float)
            floated_dfs.append(df)
        result = pd.concat(floated_dfs)
    return result

def make_date_time_col2(df):
    
    df['Hour Number'] = df_total['Hour Number'].replace(24, 0)
    df['Hour Number'] = df_total['Hour Number'].replace(25, 0)
    df['Data Date']= df['Data Date'].astype(str)
    df['Hour Number'] = df['Hour Number'].astype(str)
    df['New_datetime'] = df['Data Date'].map(str) + " " + df['Hour Number']
    df['Hour Number'] = df['Hour Number'].astype(int)
    df['New_datetime']= df['New_datetime'].apply(lambda x:f'{x}:00:00')
    df['New_datetime'] = pd.to_datetime(df['New_datetime'],infer_datetime_format=True, format ='%m/%d/%Y %H')
    df['Demand Delta'] = df['Demand Forecast (MW)']- df['Demand (MW)']
    df['Net Generation Delta'] = df['Net Generation (MW)']- df['Demand (MW)']
    print('done')
    return df

def drop_extra_cols(df_texas):
    del df_texas['UTC Time at End of Hour']
    del df_texas['Balancing Authority']
    del df_texas['Net Generation (MW) (Imputed)']
    del df_texas['Demand (MW) (Imputed)']
    del df_texas['Net Generation (MW) from All Petroleum Products']
    del df_texas['Net Generation (MW) from Unknown Fuel Sources']
    del df_texas['Data Date']
    del df_texas['Hour Number']
    del df_texas['Local Time at End of Hour']
    return df_texas

def make_hourly_demand_means2(df,lst=np.arange(0,24)):
    
    d = {}
    for i in lst:
        filt =df['Hour Number']==i
        d[i] = df.loc[filt]['Demand (MW)'].mean()
    return d

def city_weather_time_columns_adj(df):
    df['new_hour_date'] = df['hour'] + ' '+  df['Date']
    df['New_datetime'] = pd.to_datetime(df['new_hour_date'],infer_datetime_format=True, format ='%m/%d/%Y %H')
    df['time_rounded'] = df['New_datetime'].dt.round('H').dt.hour
    df['time_rounded'] = df['time_rounded'].apply(str)
    df['time_rounded2'] = df['Date'] + ' '+  df['time_rounded']
    df['time_rounded4']= df['time_rounded2'].apply(lambda x:f'{x}:00:00')
    df['New_datetime2'] = pd.to_datetime(df['time_rounded4'],infer_datetime_format=True,format ='%m/%d/%Y %H')
    df['New_datetime'] = pd.to_datetime(df['New_datetime'],infer_datetime_format=True,format ='%m/%d/%Y %H')
    return df

def city_weather_drop_cols(df):
    del df['hour']
    del df['Date']
    del df['new_hour_date']
    del df['New_datetime2']
    del df['time_rounded']
    del df['time_rounded2']
    del dallas['time_rounded4']
    
    return df

def encode_clouds(df):
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
    return df

def column_convert_float(pd_series):
    lst = []
    for i in pd_series:
        lst1 = i.split('\\')
        lst.append(lst1[0])
    results = pd.Series(lst)
    return results 

def humidity_column_convert_float3(df):
    pd_series = df['humidity']
    lst = []
    for string in pd_series:
        string = string.replace(u'\xa0%','')
        lst.append(string)
    results = pd.Series(lst)
    df['humdity1']= results
    df['humdity1'] = df['humdity1'].astype(float)
    return df

def press_column_convert_float3(df):
    
    pd_series =df['pressure']
    lst = []
    for string in pd_series:
        string = string.replace(u'\xa0in','')
        if string == '0.00\xa0':
            string = '0.00'
        lst.append(string)
    results = pd.Series(lst)
    df['pressure1']= results
    df['pressure1'] = df['pressure1'].astype(float)
    return df

def temp_column_convert_float3(df):
    pd_series = df['temp']
    lst = []
    for string in pd_series:
        string = string.replace(u'\xa0F','')
        lst.append(string)
    results = pd.Series(lst)
    df['temp1']= results
    df['temp1'] = df['temp1'].astype(float)
    return df

def wind_column_convert3(df):
    pd_series = df['wind_speed']
    lst = []
    for string in pd_series:
        string = string.replace(u'\xa0mph','')
        if string == '0.00\xa0':
            string = '0.00'
        lst.append(string)
    results = pd.Series(lst)
    df['wind1']= results
    df['wind1'] = df['wind1'].astype(float)
    return df

def precip_column_convert_float3(df):
    
    pd_series = df['precip']
    lst = []
    for string in pd_series:
        string = string.replace(u'\xa0in','')
        lst.append(string)
    results = pd.Series(lst)
    df['precip1']= results
    df['precip1'] = df['precip1'].astype(float)
    return df





if __name__ == '__main__':
    #use version 3.0 on cleans

    data = pd.read_csv('~/Downloads/EIA930_BALANCE_2020_Jan_Jun.csv')
    data_2 = pd.read_csv('~/Downloads/EIA930_BALANCE_2020_Jul_Dec.csv')

    data = data[data['Balancing Authority'] == 'ERCO']
    data_2 = data[data['Balancing Authority'] == 'ERCO']

