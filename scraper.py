import numpy as np
import pandas as pd 
import time
from datetime import datetime, timedelta
import logging


#-------------------------------------------------------------------------------------------------------------


### Functions that will be used to transform the scraped data
month_to_text = dict(zip(range(1, 13), ["January", "February", "March", "April", "May", "June", 
                              "July", "August", "Semptember", "October", "November", "December"]))


def F_to_C(t):
    return int(round((t - 32)*(5/9), 0))


def get_temperature(table):
    temperatures = table[(table['Time'].str.contains(r"^\d{1,2}:\d{2} (AM|PM)$", regex=True)) &
              (table['Temp.'].str.contains(r"^\d{1,2}(\.\d)?°F$", regex=True))]
    
    temp = temperatures['Temp.'].apply(func=lambda x: float(x[:-2]))
    
    return F_to_C(int(((np.min(temp) + np.max(temp))/2).round()))



def get_weather(table):
    temperatures = []
    
    for i in range(len(table)):
        url = f"https://weatherspark.com/h/d/98906/{table['Date'].iloc[i].year}/{table['Date'].iloc[i].month}/{table['Date'].iloc[i].day}/Historical-Weather-on-Friday-{month_to_text[table['Date'].iloc[i].month]}-{table['Date'].iloc[i].day}-{table['Date'].iloc[i].year}-in-Amman-Jordan#Figures-Temperature"
        try:
            main_table = pd.read_html(url)[3]

            temperatures.append(get_temperature(main_table))

            time.sleep(1.5001)
        except KeyError:
            main_table = pd.read_html(url)[2]
            
            temperatures.append(get_temperature(main_table))

            time.sleep(1.5001)
        except Exception as e:
            temperatures.append(np.nan)
            
    return temperatures
###


#-------------------------------------------------------------------------------------------------------------


def main():
    ### Preparing the Date column
    dates = []
    start_date = datetime(year=2001, month=1, day=1)
    end_date = datetime(year=2024, month=4, day=30)

    while start_date <= end_date:
        dates.append(start_date)
        start_date += timedelta(days=1)

    daily_data = pd.DataFrame({'Date': dates, 'Temperature': np.nan})
    ###



    ### Scraping the data and completing the dataframe
    logging.info("Starting the scraping process.")
    temperature_data = get_weather(daily_data)
    logging.info("Scraping completed.")

    daily_data['Temperature'] = temperature_data
    ###


    #-------------------------------------------------------------------------------------------------------------


    ### Exporting the data to a CSV file
    daily_data.to_csv("weather_data.csv", index=False)
    logging.info("Data exported to 'weather_data.csv'.")
    ###



if __name__ == "__main__":
    main()
