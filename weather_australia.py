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
 

###############################
###dataset information tools###
###############################

###create select box for chossing between dataset link/variable descriptions###
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
    
####create checkbox for dataset info###
if st.checkbox('Dataset info'):
    st.write("First data points:")
    st.write(weather_australia_df.head(15))

    st.write("Descriptive statistics:")
    st.write(weather_australia_df.describe().T)
    
    
###create checkbox for features distribution###
show_features_distribution_plot = st.checkbox("Features distribution")

if show_features_distribution_plot:
    fig, ax = plt.subplots(figsize=[12, 12])
    weather_australia_groupby_city.hist(ax=ax, bins=10)
    plt.suptitle('Data distribution', fontsize=16)
    
    st.pyplot(fig)


###########
###plots###
###########

###create a selct box with plots for cities with most and less weather recordings###
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
    
    

###create a selct box with plots for hottest and coldest cities###

#cities with the highest maximum temperature 
top_10_hottest_cities=weather_australia_groupby_city.sort_values(by='maxtemp', ascending=False).head(10) #sort the values from highest to lowest according to the average maxtemp and take the first 10 cities


cities_with_highest_maxtemp_plot=plt.figure(figsize=(15,8))
plt.bar(top_10_hottest_cities['maxtemp'].index, top_10_hottest_cities['maxtemp'].values, color=sb.color_palette()[1])
plt.title('Cities with highest average maximum temperature 2007-2017', fontsize=16)#make the title bigger and more readable
plt.xlabel('Cities')
plt.ylabel('Average maximum temperature temperature C°')
plt.show()

#cities with the lowest min temperature 
top_10_coldest_cities=weather_australia_groupby_city.sort_values(by='mintemp', ascending=True).head(10) #sort the values from lowest to highest according to the average mintemp and take the first 10 cities

cities_with_lowest_mintemp_plot=plt.figure(figsize=(15,8))
plt.bar(top_10_coldest_cities['mintemp'].index, top_10_coldest_cities['mintemp'].values, color=sb.color_palette()[2])
plt.title('Cities with lowest average minimum temperature 2007-2017', fontsize=16)#make the title bigger and more readable
plt.xlabel('Cities')
plt.ylabel('Average minimum temperature temperature C°')
plt.show()


st.subheader('Australian hottest and coldest cities')
hottest_and_coldest_cities_plots=st.selectbox('Select:', ('Hottest cities', 'Coldest cities'))
if hottest_and_coldest_cities_plots=='Hottest cities':
    st.write(cities_with_highest_maxtemp_plot)
if hottest_and_coldest_cities_plots=='Coldest cities':
    st.write(cities_with_lowest_mintemp_plot)


###explore what are the most rainy cities###
st.subheader('What are the most rainy cities?')

weather_australia_modified=weather_australia_df.copy()
#I map the values of the 'raintoday'column No-->0 Yes-->1
weather_australia_modified['raintoday']=weather_australia_modified['raintoday'].replace({'No':0, 'Yes':1}) 
#I map the values of the 'raintomorrow'column No-->0 Yes-->1
weather_australia_modified['raintomorrow']=weather_australia_modified['raintomorrow'].replace({'No':0, 'Yes':1})
#I am interested in considering the 'locaiton' and 'raintoday' columns only 
weather_australia_groupby_city_sum_raintoday=weather_australia_modified[['location', 'raintoday']].groupby('location').sum() #I group the dataframe by location and sum the values 
#I concatenete the two Series of weather recordings and rainy days for each city
pd.concat([weather_recordings_per_city, weather_australia_groupby_city_sum_raintoday ], axis=1)
rainy_days_per_location_df= pd.concat([weather_recordings_per_city, weather_australia_groupby_city_sum_raintoday ], axis=1) #assign a name to the df
rainy_days_per_location_df.columns=['weather_recordings', 'rainy_days'] #give new names to columns
#create a new column showing the ratio between rainy days and weather recordings per city. Those city with the highest ration can be considered the most rainy cities
rainy_days_per_location_df['pct_of_rainy_days']=round(rainy_days_per_location_df['rainy_days']/rainy_days_per_location_df['weather_recordings'], 2) #round the ration to the second decimal place
top_10_cities_for_rainy_days=rainy_days_per_location_df['pct_of_rainy_days'].sort_values(ascending=False).head(10) #select the top 10 cities with highest percentage of rainy days
top_10_cities_for_drought_days=rainy_days_per_location_df['pct_of_rainy_days'].sort_values(ascending=False).tail(10) #select the top 10 cities with lowest percentage of rainy days

fig, ax =plt.subplots(1,2, figsize=(15,5))

ax[0].bar(top_10_cities_for_rainy_days.index, top_10_cities_for_rainy_days.values, width=0.5, color=sb.color_palette()[0])
ax[0].set_xlabel('Cities')
ax[0].set_xticklabels(top_10_cities_for_rainy_days.index, rotation=45)
ax[0].set_ylabel('Percentage of rainy days over total weather recordings')
ax[0].set_title('Most rainy cities')

ax[1].bar(top_10_cities_for_drought_days.index, top_10_cities_for_drought_days.values, width=0.5, color=sb.color_palette()[5])
ax[1].set_xlabel('Cities')
ax[1].set_xticklabels(top_10_cities_for_drought_days.index, rotation=45)
ax[1].set_ylabel('Percentage of rainy days over total weather recordings')
ax[1].set_ylim(0, 0.35) #set the y axis bounds in order to have both plots with the same scale
ax[1].set_title('Most drought cities')
fig.suptitle('Cities with most and less rainy days 2007-2017: prercentage of rainy days over total weather recordings', fontsize=16)#make the title bigger and more readable

st.pyplot(fig)

st.markdown("""What are the cities with highest average mm of rain?
I want to explore whether or not those cities that have the highest percentage of rainy days are also the cities with highest average mm of rain.
If not, this would mean that some cities are subject to occasional but strong rainfalls.
            """)

cities_sorted_by_mm_of_rain= weather_australia_groupby_city['rainfall'].sort_values( ascending=False)#sort the cities by their average mm of rain
#select the top 10 cities by average mm of rain
top_10_cities_for_mm_of_rain= weather_australia_groupby_city['rainfall'].sort_values( ascending=False).head(10)

#create the plot of top 10 cities with highest average mm of rain
most_rainy_cities_by_mm_of_rain_plot=plt.figure(figsize=(15,8))
plt.barh(top_10_cities_for_mm_of_rain.index,top_10_cities_for_mm_of_rain.values, color=sb.color_palette()[2])#horizontal bar plot
plt.title('Cities with highest average mm of rain 2007-2017', fontsize=16)
plt.xlabel('Average mm of rain')
plt.ylabel('Cities')

st.write(most_rainy_cities_by_mm_of_rain_plot)

st.markdown('Explore whether the most rainy city, which are the locations with the highest percentage of rainy days, are also the cities with highest averge mm of rain.')

#create plot for cities withhighest average mm of rain and percetage of rainy days
most_rainy_cities_by_mm_of_rain_and_percentage_of_rainy_days_plot=plt.figure(figsize=(25,18))
figure=plt.barh(cities_sorted_by_mm_of_rain.index, cities_sorted_by_mm_of_rain.values, color=sb.color_palette()[2])#use horizontal barchart
plt.title('Cities with highest average mm of rain 2007-2017', fontsize=30)#make the title bigger and more readable
plt.xlabel('Average mm of rain', fontsize=23)
plt.ylabel('City')
plt.yticks(fontsize=20)
#color in red those cities that belongs to the top 10 rainy cities group

for i, city in enumerate(cities_sorted_by_mm_of_rain.index): #iterate on every city and on its index
    if city in top_10_cities_for_rainy_days.index: #if the city belongs to the top 10 rainy city group, it is colred in red
        figure[i].set_color('red')#color the city according to its index poistion in top_10_cities_for_mm_of_rain.index        
 #create a legend       
plt.legend([plt.Rectangle((0,0),1,1, fc='red', edgecolor='black')], ['cities with highest percentage of rainy days'], fontsize=18)

st.write(most_rainy_cities_by_mm_of_rain_and_percentage_of_rainy_days_plot)



######################
###SIDEBAR SECTIONS###
######################

###plots fot the Best Australian cities weather section###
#distribution of maxtemp--> I choose values between 20-30 for the pleasent_maxtemp_mask
maxtemp_distribution_plot=sb.displot(weather_australia_groupby_city['maxtemp'], kde=True )
plt.suptitle('Distribution of maxtemp')

#distribution of windspeed3pm--> I choose values between 18 and 22 for the pleasent_windspeed3pm_mask
windspeed3pm_distribution_plot=sb.displot(weather_australia_groupby_city['windspeed3pm'], kde=True )
plt.suptitle('Distribution of windspeed3pm')

#distribution of rainfall--> I choose values between 2 and 3 for the pleasent_rainfall_mask
rainfall_distribution_plot=sb.displot(weather_australia_groupby_city['rainfall'], kde=True )
plt.suptitle('Distribution of rainfall')

#create the masks:
pleasent_maxtemp_mask= (weather_australia_groupby_city['maxtemp']>20) & (weather_australia_groupby_city['maxtemp']<30)
pleasent_windspeed3pm_mask= (weather_australia_groupby_city['windspeed3pm']>18) & (weather_australia_groupby_city['windspeed3pm']<22)
pleasent_rainfall_mask= (weather_australia_groupby_city['rainfall']>2) & (weather_australia_groupby_city['rainfall']<3)

#scatterplot
best_australian_cities_for_weather_scatterplot=plt.figure(figsize=(15, 8))
plt.scatter(weather_australia_groupby_city['maxtemp'].values, weather_australia_groupby_city['rainfall'].values, label='other cities')
plt.scatter(weather_australia_groupby_city.loc['Albany']['maxtemp'], weather_australia_groupby_city.loc['Albany']['rainfall'], color='red', marker='*', label='Albany') #plot the values of maxtemp and rainfall of Albany using .loc() method since the index are the cities' names; make the dot more visible with marker = *
plt.scatter(weather_australia_groupby_city.loc['Witchcliffe']['maxtemp'], weather_australia_groupby_city.loc['Witchcliffe']['rainfall'], color='orange', marker='*', label='Withcliffe') #plot the values of maxtemp and rainfall of Withcliffe using .loc() method since the index are the cities' names; make the dot more visible with marker = *
plt.xlabel('Average maxtemp C°')
plt.ylabel('Avergae rainfall mm')
plt.title('Cities with most pleasent weather', fontsize=16)#make the title bigger and more readable
plt.legend()
plt.xticks(rotation=45)


def cities_with_most_pleasent_weather():
    st.title("Best Australian cities for weather")
    st.markdown("""
             The technique of deciding which Australian city has the nicest weather is arbitrary.
             Nonetheless, highlighting the cities with neither extremely high nor extremely low temperatures,
             precipitation, or wind speed might provide some interesting information.
             """)
    st.markdown("""
                In order to evaluate the nicest city in terms of weather I consider 3 variables:

* the maximum temperature
* wind speed in the afternoon
* the average mm of rain 

                """)
    st.markdown(""" 
                The values of the tempearture, wind speed and rain that I am going to consider as 'more pleasent' are values close to the mean.
                In order to see what are the values close to the mean, it is usefull to recall the data distribution plots of the variables.
                """)
    st.pyplot(maxtemp_distribution_plot)
    st.pyplot(windspeed3pm_distribution_plot)
    st.pyplot(rainfall_distribution_plot)
    
    st.markdown("""
                Only 2 cities meet my requests for an optimal climate:
* Albany
* Withcliffe

Let's have an overview on the climate of these cities in comparison with other cities:
                              """)
    st.write(best_australian_cities_for_weather_scatterplot)
    
    
    



def climate_change_2007_2017():
    st.title("Climate change 2007- 2017")
    st.write("SECTIOS")








def extreme_weather_events():
    st.title("Extreame weather events")
    st.write("SECTIONSSSS")

# Sidebar
st.sidebar.title("Sections")

# Pulsanti per le pagine
best_city_for_weather_page= st.sidebar.button("Best Australian cities for weather")
climate_change_page= st.sidebar.button("Climate change 2007- 2017")
extreme_weather_events_page= st.sidebar.button("Extreame weather events")

# Logica per visualizzare la pagina corretta
if best_city_for_weather_page:
    cities_with_most_pleasent_weather()
elif climate_change_page:
    climate_change_2007_2017()
elif extreme_weather_events_page:
    extreme_weather_events()
    

    

#create a sidebar about extreme weather events





#create a sidebar about climate change 2007-2017



#create a sidebar about cities with most pleasent weather

