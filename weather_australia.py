#import all the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


#imprt stramlit
import streamlit as st

weather_australia_df=pd.read_csv('weather_australia.csv')

st.set_page_config(layout='centered')
st.header('Australia weather')
st.subheader('weather recordings of Australian cities from 2007 to 2017')
 
st.subheader('Go to dataset link/ variables descriptions')
curtain= st.selectbox('select:', ('Dataset link', 'Variables description'))
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
    
    
if st.checkbox('Dataset info'):
    st.write("Primi 15 record del dataset:")
    st.write(weather_australia_df.head(15))

    st.write("Descrizione del dataset:")
    st.write(weather_australia_df.describe().T)
    
###################### 
###cleaning dataset###
######################
#copia tutte le celle per la pulizia



