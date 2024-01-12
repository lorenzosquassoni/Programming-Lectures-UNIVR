#import all the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


#import stramlit
import streamlit as st

#Import a new palette for colored plots
sb.set_palette("Set2")
sb.color_palette("Set2")

weather_australia_df=pd.read_csv('weather_australia.csv')

st.set_page_config(layout='centered')
st.header('Australia weather')
st.subheader('weather recordings of Australian cities from 2007 to 2017')
 
 
###################### 
###cleaning dataset###
######################

#rename all the columns  with all lower case letters, and remove possible empty spaces at the beginning and end od the column name
weather_australia_df.columns= list(map(lambda x:x.lower().strip(), weather_australia_df.columns))

#the column is a str type so I cast into datetime
weather_australia_df['date'].dtype
weather_australia_df['date']= pd.to_datetime(weather_australia_df['date'], format='%Y-%m-%d')

#I group the data points by city in order to see if some cities have no data for specific columns.

weather_australia_groupby_city=weather_australia_df.groupby('location').mean(numeric_only=True)

#drop useless columns
#evaporation column has too many null values and is not usefull for analysing the climate in Australia from 2007 to 2017
#windgustdir column is not usefull for analysing the climate in Australia from 2007 to 2017
#sunshine column has too many null values and is not usefull for analysing the climate in Australia from 2007 to 2017
#cloud9am column has too many null values and is not usefull for analysing the climate in Australia from 2007 to 2017
##cloud3am has too many null values and is not usefull for analysing the climate in Australia from 2007 to 2017
#winddir9pm is not usefull for analysing the climate in Australia from 2007 to 2017
#winddir3pm is not usefull for analysing the climate in Australia from 2007 to 2017

weather_australia_df.drop(['evaporation', 'windgustdir', 'sunshine', 'cloud9am', 'cloud3pm', 'winddir9am', 'winddir3pm'], axis=1, inplace=True)

#I replace the null values of the 'mintemp' variable with the mean of mintemp for each specific city
weather_australia_df['mintemp']= weather_australia_df['mintemp'].fillna(weather_australia_df.groupby('location')['mintemp'].transform('mean'))


#Fixing 'maxtemp' null values: I apply the same tecnique as before, I replace the null values with the mean of the correponding city
weather_australia_df['maxtemp']= weather_australia_df['maxtemp'].fillna(weather_australia_df.groupby('location')['maxtemp'].transform('mean'))

#the missing values in 'raintoday' column are replaced with boolean value 'No' which means that precipitation has been less than 1 mm
weather_australia_df['raintoday'].fillna('No', inplace=True)

#the missing values in 'rainfall' column are replaced with value zero
weather_australia_df['rainfall'].fillna(0.0, inplace=True)

#Fixing 'windgustspeed' null values: for the column concerning the strongest wind gust during the day is not possible to replace the null values with the mean of the correponding city becuase for Albany and Newcastle all values for 'windgustspeed' are null.the null valus can only be replaced with the mean of all other cities.
#I replace the null values with the mean of the entire dataset
weather_australia_df['windgustspeed']=weather_australia_df['windgustspeed'].fillna(weather_australia_df['windgustspeed'].mean())

#Fixing 'windspeed9am' null values: I replace the null values with the mean of the correponding city
weather_australia_df['windspeed9am']= weather_australia_df['windspeed9am'].fillna(weather_australia_df.groupby('location')['windspeed9am'].transform('mean'))

#Fixing 'windspeed3pm' null values: I replace the null values with the mean of the correponding city
weather_australia_df['windspeed3pm']= weather_australia_df['windspeed3pm'].fillna(weather_australia_df.groupby('location')['windspeed3pm'].transform('mean'))

#Fixing 'humidity9am' null values: I replace the null values with the mean of the correponding city
weather_australia_df['humidity9am']= weather_australia_df['humidity9am'].fillna(weather_australia_df.groupby('location')['humidity9am'].transform('mean'))

#Fixing 'humidity3pm' null values: I replace the null values with the mean of the correponding city
weather_australia_df['humidity3pm']= weather_australia_df['humidity3pm'].fillna(weather_australia_df.groupby('location')['humidity3pm'].transform('mean'))

#Fixing 'pressure9am' null values: the null valus can only be replaced with the mean of all other cities.
weather_australia_df['pressure9am']=weather_australia_df['pressure9am'].fillna(weather_australia_df['pressure9am'].mean())

#Fixing 'pressure3pm' null values: the null valus can only be replaced with the mean of all other cities.
weather_australia_df['pressure3pm']=weather_australia_df['pressure3pm'].fillna(weather_australia_df['pressure3pm'].mean())

#Fixing 'temp9am' null values: I replace the null values with the mean of the correponding city
weather_australia_df['temp9am']= weather_australia_df['temp9am'].fillna(weather_australia_df.groupby('location')['temp9am'].transform('mean'))

#Fixing 'temp3pm' null values: I replace the null values with the mean of the correponding city
weather_australia_df['temp3pm']= weather_australia_df['temp3pm'].fillna(weather_australia_df.groupby('location')['temp3pm'].transform('mean'))

#Fixing 'raintomorrow' null values: I replace the null values with boolean Value 'No' meaning that the amount of rain of the next day is less than 1 mm
weather_australia_df['raintomorrow'].fillna('No', inplace=True)

#Some null values are still present in the dataframe in which weather recordings are grouped by city
#I drop the useless columns as I did for the ungrouped dataframe
weather_australia_groupby_city.drop(['evaporation', 'sunshine', 'cloud9am', 'cloud3pm'], axis=1, inplace=True)

#cities with null values for 'windgustspeed'
weather_australia_groupby_city[weather_australia_groupby_city.isnull()['windgustspeed']].windgustspeed
#cities with null values for 'pressure9am'
weather_australia_groupby_city[weather_australia_groupby_city.isnull()['pressure9am']].pressure9am
#cities with null values for 'pressure3pm'
weather_australia_groupby_city[weather_australia_groupby_city.isnull()['pressure3pm']].pressure3pm

#replace null values for the column 'windgustspeed
weather_australia_groupby_city['windgustspeed']=weather_australia_groupby_city['windgustspeed'].fillna(weather_australia_groupby_city['windgustspeed'].mean())

#replace null values for the column 'pressure9am'
weather_australia_groupby_city['pressure9am']=weather_australia_groupby_city['pressure9am'].fillna(weather_australia_groupby_city['pressure9am'].mean())

##replace null values for the column 'pressure3pm'
weather_australia_groupby_city['pressure3pm']=weather_australia_groupby_city['pressure3pm'].fillna(weather_australia_groupby_city['pressure3pm'].mean()) 
 
 
#create select box for chossing between dataset link/variable descriptions
st.subheader('Go to dataset link/ variables descriptions')
curtain= st.selectbox('Select:', ('Dataset link', 'Variables description'))
if curtain=='Dataset link':
    st.write('https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package')
if curtain== 'Variables description':
    st.markdown("""
                * Location: the common name of the location of the weather station
* MinTemp: the minimum temperature in degrees celsius
* MaxTemp: the maximum temperature in degrees celsius
* Rainfall: the amount of rainfall recorded for the day in mm
* Evaporation: evaporation (mm) in the 24 hours to 9am 
* Sunshine: the number of hours of bright sunshine in the day 
* WindGusDir: the direction of the strongest wind gust in the 24 hours to midnight
* WindGuSpeed: the speed (km/h) of the strongest wind gust in the 24 hours to midnight
* WindDir9am: direction of the wind at 9am
* WindDir3pm: direction of the wind at 3pm
* WindSpeed9am: wind speed (km/hr) averaged over 10 minutes prior to 9am
* WindSpeed3pm: wind speed (km/hr) averaged over 10 minutes prior to 3pm
* Humidity9am: humidity (percent) at 9am
* Humidity3pm: humidity (percent) at 3pm
* Pressure9am: atmospheric pressure (hpa) reduced to mean sea level at 9am
* Pressure3pm: atmospheric pressure (hpa) reduced to mean sea level at 3pm
* Cloud9am:  fraction of sky obscured by cloud at 9am. This is measured in "oktas", which are a unit of eigths. It records how many
* Cloud3pm: fraction of sky obscured by cloud (in "oktas": eighths) at 3pm. See Cload9am for a description of the values
* Temp9am: temperature (degrees C) at 9am
* Temp3pm: temperature (degrees C) at 3pm
* RainToday: boolean: 1 if precipitation (mm) in the 24 hours to 9am exceeds 1mm, otherwise 0
* RainTomorrow: the amount of next day rain in mm. 
                """)
    
#create checkbox for dataset info    
if st.checkbox('Dataset info'):
    st.write("First data points:")
    st.write(weather_australia_df.head(15))

    st.write("Descriptive statistics:")
    st.write(weather_australia_df.describe().T)
    
    
#create checkbox for features distribution
show_features_distribution_plot = st.checkbox("Features distribution")

if show_features_distribution_plot:
    fig, ax = plt.subplots(figsize=[12, 12])
    weather_australia_groupby_city.hist(ax=ax, bins=10)
    plt.suptitle('Data distribution', fontsize=16)
    
    st.pyplot(fig)



#create a selct box with plots for cities with most and less weather recordings
weather_recordings_per_city=weather_australia_df['location'].value_counts()
cities_with_most_weather_recordings=weather_recordings_per_city.head(10)
cities_with_less_weather_recordings=weather_recordings_per_city.tail(10)


cities_with_most_weather_recordings_plot=plt.figure(figsize=(15,8))
plt.bar(cities_with_most_weather_recordings.index, cities_with_most_weather_recordings.values, width= 0.5, color=sb.color_palette()[0])
plt.xlabel('Cities')
plt.xticks(cities_with_most_weather_recordings.index, rotation=45)
plt.ylabel('Number of weather recordings')
plt.title('Cities with most weather recordings')

cities_with_less_weather_recordings_plot=plt.figure(figsize=(15,8))
plt.bar(cities_with_less_weather_recordings.index, cities_with_less_weather_recordings.values, width= 0.5, color=sb.color_palette()[1])
plt.xlabel('Cities')
plt.xticks(cities_with_less_weather_recordings.index, rotation=45)
plt.ylabel('Number of weather recordings')
plt.title('Cities with less weather recordings')

st.subheader('Cities with less and most weather recordings')
weather_recordings_plots=st.selectbox('Select:', ('Cities with most weather recordings', 'Cities with less weather recordings'))
if weather_recordings_plots=='Cities with most weather recordings':
    st.write(cities_with_most_weather_recordings_plot)
if weather_recordings_plots=='Cities with less weather recordings':
    st.write(cities_with_less_weather_recordings_plot)

