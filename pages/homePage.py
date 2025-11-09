import streamlit as st
from streamlit_calendar import calendar
import pandas as pd
import numpy as np
# from datetime import date

# home page elements
DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
         'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

@st.cache_data
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')

# Load 10,000 rows of data into the dataframe.
data = load_data(10000)

# Notify the reader that the data was successfully loaded.
data_load_state.text("")

st.subheader('Mood Graph:')
hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
st.line_chart(hist_values,x_label="Time",y_label="Mood",color="#64C46C")

# update to track current day - update dependencies
filtered_data = data[data[DATE_COLUMN].dt.day==9]
st.subheader("Map of your activity today:")
st.map(filtered_data)
