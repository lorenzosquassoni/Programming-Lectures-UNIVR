#import all the libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sb
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.title('_Weather Australia 2007-2017_')
weather_australia_df=pd.read_csv('weather_australia.csv')
weather_australia_df
 
st.write(weather_australia_df.head(10))
st.write(weather_australia_df)
