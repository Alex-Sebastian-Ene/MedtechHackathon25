import streamlit as st
from streamlit_geolocation import geo
import pandas as pd
import time

#TO DO: Extract HOME_SET state variable from database

#dataframe
st.title("Places you've been")


current_loc = geo()

#location successfully retrieved?
if current_loc:

    latitude = location.get('latitude')
    longitude = location.get('longitude')

    if latitude and longitude:
        #update location
        #TO DO : upload current_loc data to SQL database
        time.sleep(60)


#show map

