from bs4 import BeautifulSoup as BS
from selenium import webdriver
from functools import reduce
import pandas as pd
import time
import xport
import pandas as pd

def render_page(url):
        driver = webdriver.Chrome('/Users/cp/Downloads/chromedriver')
        driver.get(url)
        time.sleep(3)
        r = driver.page_source
        driver.quit()
        return r
def scraper2(page, dates):
    output = pd.DataFrame()

    for d in dates:

        url = str(str(page) + str(d))

        r = render_page(url)

        soup = BS(r, "html.parser")
        container = soup.find('lib-city-history-observation')
        check = container.find('tbody')

        data = []

        for c in check.find_all('tr', class_='ng-star-inserted'):
            for i in c.find_all('td', class_='ng-star-inserted'):
                trial = i.text
                trial = trial.strip('  ')
                data.append(trial)

        if len(data)%2 == 0:
            hour = pd.DataFrame(data[0::10], columns = ['hour'])
            temp = pd.DataFrame(data[1::10], columns = ['temp'])
            dew = pd.DataFrame(data[2::10], columns = ['dew'])
            humidity = pd.DataFrame(data[3::10], columns = ['humidity'])
            wind_speed = pd.DataFrame(data[5::10], columns = ['wind_speed'])
            pressure =  pd.DataFrame(data[7::10], columns = ['pressure'])
            precip =  pd.DataFrame(data[8::10], columns = ['precip'])
            cloud =  pd.DataFrame(data[9::10], columns = ['cloud'])
            
        dfs = [hour, temp,dew,humidity,  wind_speed,  pressure, precip, cloud]
        df_final = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), dfs)
        
        df_final['Date'] = str(d) + "-" + df_final.iloc[:, :1].astype(str)
        
        output = output.append(df_final)

    print('Scraper done!')
    output = output[['hour', 'temp', 'dew', 'humidity', 'wind_speed', 'pressure',
                        'precip', 'cloud']]
    
    return output
def jan_dates():
    lst = []
    for i in range(1, 32):
        i = str(i)
        lst.append(str("2021-1-"+i))
    return lst
january = jan_dates()

def feb_dates():
    lst = []
    for i in range(1, 30):
        i = str(i)
        lst.append(str("2020-2-"+i))
    return lst
feb = feb_dates()
def march_dates():
    lst = []
    for i in range(1, 32):
        i = str(i)
        lst.append(str("2020-3-"+i))
    return lst
mar = march_dates()

def april_dates():
    lst = []
    for i in range(1, 31):
        i = str(i)
        lst.append(str("2020-4-"+i))
    return lst
april = april_dates()

def may_dates():
    lst = []
    for i in range(1, 32):
        i = str(i)
        lst.append(str("2020-5-"+i))
    return lst
may = may_dates()

def june_dates():
    lst = []
    for i in range(1, 31):
        i = str(i)
        lst.append(str("2020-6-"+i))
    return lst
june = june_dates()

def july_dates():
    lst = []
    for i in range(1, 32):
        i = str(i)
        lst.append(str("2020-7-"+i))
    return lst
july = july_dates()

def august_dates():
    lst = []
    for i in range(1, 32):
        i = str(i)
        lst.append(str("2020-8-"+i))
    return lst
august = august_dates()

def september_dates():
    lst = []
    for i in range(1, 31):
        i = str(i)
        lst.append(str("2020-9-"+i))
    return lst
september = september_dates()

def october_dates():
    lst = []
    for i in range(1, 32):
        i = str(i)
        lst.append(str("2020-10-"+i))
    return lst
october = october_dates()

def november_dates():
    lst = []
    for i in range(1, 8):
        i = str(i)
        lst.append(str("2020-11-"+i))
    return lst
november_to7 = november_dates()

def november_dates_end():
    lst = []
    for i in range(9, 31):
        i = str(i)
        lst.append(str("2020-11-"+i))
    return lst
november_end = november_dates_end()

def december_dates():
    lst = []
    for i in range(1, 32):
        i = str(i)
        lst.append(str("2020-12-"+i))
    return lst
december = december_dates()

def scraper3(page, dates):
    output = pd.DataFrame()

    for d in dates:

        url = str(str(page) + str(d))

        r = render_page(url)

        soup = BS(r, "html.parser")
        container = soup.find('lib-city-history-observation')
        check = container.find('tbody')

        data = []

        for c in check.find_all('tr', class_='ng-star-inserted'):
            for i in c.find_all('td', class_='ng-star-inserted'):
                trial = i.text
                trial = trial.strip('  ')
                data.append(trial)

        if len(data)%2 == 0:
            hour = pd.DataFrame(data[0::10], columns = ['hour'])
            temp = pd.DataFrame(data[1::10], columns = ['temp'])
            dew = pd.DataFrame(data[2::10], columns = ['dew'])
            humidity = pd.DataFrame(data[3::10], columns = ['humidity'])
            wind_speed = pd.DataFrame(data[5::10], columns = ['wind_speed'])
            pressure =  pd.DataFrame(data[7::10], columns = ['pressure'])
            precip =  pd.DataFrame(data[8::10], columns = ['precip'])
            cloud =  pd.DataFrame(data[9::10], columns = ['cloud'])
#                 date2 = pd.DataFrame(str(str(d)), columns = ['date2'])
            
        dfs = [hour, temp,dew,humidity,  wind_speed,  pressure, precip, cloud]
        df_final = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), dfs)
        
        df_final['Date'] = str(d)
        
        output = output.append(df_final)

    print('Scraper done!')
    output = output[['hour', 'temp', 'dew', 'humidity', 'wind_speed', 'pressure',
                        'precip', 'cloud', 'Date']]
#         output.to_csv (f"r'/Users/cp/Desktop/capstone2/{dates[0]}_scraped_temps.csv'", index = False, header=True)
    return output
#         return data

def weather_scrape(lst2):
    months = []
    for i in lst2:
        month1 = scraper3(page, i)
        months.append(month1)
    year =  pd.concat(months)
    return year
# year_2020 = weather_scrape(lst)
# year_2020.to_csv (r'/Users/cp/Desktop/capstone2/DALLAS_YEAR_SCRAPE.csv', index = False, header=True)




if __name__ =='__main__':

    #page = 'https://www.wunderground.com/history/daily/us/tx/dallas/KDAL/date/'
  
  def scraper(page, dates):
        output = pd.DataFrame()

        for d in dates:

            url = str(str(page) + str(d))

            r = render_page(url)

            soup = BS(r, "html.parser")
            container = soup.find('lib-city-history-observation')
            check = container.find('tbody')

            data = []

            for c in check.find_all('tr', class_='ng-star-inserted'):
                for i in c.find_all('td', class_='ng-star-inserted'):
                    trial = i.text
                    trial = trial.strip('  ')
                    data.append(trial)

            if round(len(data) / 10) == 23:
                hour = pd.DataFrame(data[0::10], columns = ['hour'])
                temp = pd.DataFrame(data[1::10], columns = ['temp'])
                dew = pd.DataFrame(data[2::10], columns = ['dew'])
                humidity = pd.DataFrame(data[3::10], columns = ['humidity'])
                wind_speed = pd.DataFrame(data[5::10], columns = ['wind_speed'])
                pressure =  pd.DataFrame(data[7::10], columns = ['pressure'])
                precip =  pd.DataFrame(data[8::10], columns = ['precip'])
                cloud =  pd.DataFrame(data[9::10], columns = ['cloud'])
                
            dfs = [hour, temp,dew,humidity,  wind_speed,  pressure, precip, cloud]
            df_final = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), dfs)
            
            df_final['Date'] = str(d) + "-" + df_final.iloc[:, :1].astype(str)
            
            output = output.append(df_final)

        print('Scraper done!')
        output = output[['hour', 'temp', 'dew', 'humidity', 'wind_speed', 'pressure',
                         'precip', 'cloud']]
        
        return output
#                
#             else:
#                 print('Data not in normal length')

#             dfs = [Date, Temperature, Dew_Point, Humidity, Wind, Pressure, Precipitation]

#             df_final = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), dfs)

#             df_final['Date'] = str(d) + "-" + df_final.iloc[:, :1].astype(str)

#             output = output.append(df_final)

#         print('Scraper done!')

#         output = output[['Temp_avg', 'Temp_min', 'Dew_max', 'Dew_avg', 'Dew_min', 'Hum_max',
#                          'Hum_avg', 'Hum_min', 'Wind_max', 'Wind_avg', 'Wind_min', 'Pres_max',
#                          'Pres_avg', 'Pres_min', 'Precipitation', 'Date']]

#         return output
        return data