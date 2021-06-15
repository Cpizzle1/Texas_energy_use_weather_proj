import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

def weather_clean(df):
        df['new_hour_date'] = df['hour'] + ' '+  df['Date']
        df['New_datetime'] = pd.to_datetime(df['new_hour_date'],infer_datetime_format=True, format ='%m/%d/%Y %H')
        df['time_rounded'] = df['New_datetime'].dt.round('H').dt.hour
        df['time_rounded'] = df['time_rounded'].apply(str)
        df['time_rounded2'] = df['Date'] + ' '+  df['time_rounded']
        df['time_rounded4']= df['time_rounded2'].apply(lambda x:f'{x}:00:00')
        df['New_datetime2'] = pd.to_datetime(df['time_rounded4'],infer_datetime_format=True,format ='%m/%d/%Y %H')
        
        df['New_datetime']= df['time_rounded4']
        df['New_datetime'] = pd.to_datetime(df['New_datetime'],infer_datetime_format=True,format ='%m/%d/%Y %H')
        del df['hour']
        del df['Date']
        del df['new_hour_date']
        del df['New_datetime']
        del df['time_rounded']
        del df['time_rounded2']
        # del dallas['time_rounded3']
        del df['time_rounded4']
        return df



if __name__ == '__main__':

    
    dallas = pd.read_csv('~/Desktop/Dallas_YEAR_SCRAPE.csv')
    dallas_clean =  weather_clean(dallas)
    print(dallas_clean)

