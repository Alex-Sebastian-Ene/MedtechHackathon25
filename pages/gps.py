import streamlit as st
from streamlit_geolocation import streamlit_geolocation as geo
import pandas as pd
from datetime import datetime, date, timedelta
import pydeck as pdk
import sys
from pathlib import Path

# add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from databases.database import get_connection, initialize_database

initialize_database()
#queries to upload to database


LOC_UPDATE_SESSIONS_QUERY = """
INSERT INTO gps_sessions (user_id, started_at)
VALUES (:userid, :timestamp);
"""
LOC_GET_SESSION_QUERY = """
SELECT session_id 
FROM gps_sessions
WHERE user_id = :userid
ORDER BY started_at DESC
LIMIT 1;
"""

LOC_UPDATE_POINTS_QUERY = """
INSERT INTO gps_points (session_id, recorded_at, latitude, longitude)
VALUES (:session_id, :timestamp, :lat, :long);
"""

LOC_DOWNLOAD_QUERY = """
SELECT gp.longitude, gp.latitude, gp.recorded_at
FROM gps_points gp
JOIN gps_sessions gs ON gp.session_id = gs.session_id
WHERE gs.user_id = :userid 
  AND gp.recorded_at >= :date_scope
ORDER BY gp.recorded_at ASC;
"""

st.title("Places you've been")

userid = st.session_state['user_id']
current_loc = geo()
timestamp = datetime.now()
latitude = False
longitude = False
session_id = []

with get_connection() as conn:
    session_id = conn.execute(LOC_GET_SESSION_QUERY, {"userid": userid}).fetchall()

#location successfully retrieved?
if current_loc:

    latitude = current_loc.get('latitude')
    longitude = current_loc.get('longitude')

    if latitude and longitude:
        #update location
        #TO DO : upload current_loc data to SQL database
        
        with get_connection() as conn:  # ✅ added ()
            conn.execute(LOC_UPDATE_SESSIONS_QUERY, {"userid": userid, "timestamp": timestamp})
            
            #session_id[-1] returns most recent sessionid 
            conn.execute(LOC_UPDATE_POINTS_QUERY, {"session_id": session_id[-1], "timestamp": timestamp, "lat": latitude, "long": longitude})

#date scopes
today = datetime.combine(date.today(), datetime.min.time())
now = datetime.now()
week = now - timedelta(days=now.weekday())  # weekday() → Monday=0, Sunday=6
week = week.replace(hour=0, minute=0, second=0, microsecond=0)
month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

date_scope = {"Past 24hrs": today, "Past Week": week, "Past Month": month}

scope = st.select_slider(
    "Select a time period",
    options=[
        "Past 24hrs",
        "Past Week",
        "Past Month",
    ]
)

#download data
map_points = []

with get_connection() as conn:  # ✅ added ()
    map_points = conn.execute(LOC_DOWNLOAD_QUERY, {"userid": userid, "date_scope": date_scope[scope]}).fetchall()  # ✅ fetchall()

# Convert to DataFrame
df = pd.DataFrame(map_points, columns=['lon', 'lat'])

# Soft, calming colors for markers
marker_color = [100, 149, 237]  # Soft blue (Cornflower Blue)

# Create a gentle ScatterplotLayer
layer = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position='[lon, lat]',
    get_color=marker_color,
    get_radius=500,
    pickable=True,
    auto_highlight=True
)

view_state = pdk.ViewState(
    longitude=df['lon'].mean(),
    latitude=df['lat'].mean(),
    zoom=10,
    pitch=0
)

r = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    map_style='mapbox://styles/mapbox/light-v10',
)

st.pydeck_chart(r)
