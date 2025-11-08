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
FROM gps_points AS gp
JOIN gps_sessions AS gs ON gp.session_id = gs.session_id
WHERE gs.user_id = :userid 
  AND gp.recorded_at >= :date_scope
ORDER BY gp.recorded_at ASC
"""

# Login protection
if 'user_id' not in st.session_state:
    st.warning("Please log in to access this page.")
    st.switch_page("app.py")
    st.stop()

st.title("Places you've been")
st.write(f"Welcome, {st.session_state.get('username', 'User')}!")

userid = st.session_state['user_id']

# Button to trigger location refresh
location_button = st.button("Refresh My Location", key="get_location")

if location_button:
    st.rerun()

current_loc = geo()
print (current_loc)

timestamp = datetime.now().isoformat()
latitude = None
longitude = None

# Debug: Show raw location data
with st.expander("DEBUG - Raw Location Data", expanded=False):
    st.json(current_loc)
    st.write(f"Type of current_loc: {type(current_loc)}")
    st.write(f"Is dict: {isinstance(current_loc, dict)}")
    if current_loc:
        st.write(f"Keys: {current_loc.keys() if isinstance(current_loc, dict) else 'N/A'}")

# Display current location status
st.write(f"DEBUG Step 1: current_loc exists = {bool(current_loc)}, is dict = {isinstance(current_loc, dict)}")

if current_loc and isinstance(current_loc, dict):
    latitude = current_loc.get('latitude')
    longitude = current_loc.get('longitude')
    
    st.write(f"DEBUG Step 2: Parsed - Lat={repr(latitude)} (type={type(latitude).__name__}), Long={repr(longitude)} (type={type(longitude).__name__})")
    
    # The library might return None - let's check if they are valid
    lat_valid = latitude is not None and latitude != 0
    long_valid = longitude is not None and longitude != 0
    
    st.write(f"DEBUG Step 3: lat_valid={lat_valid}, long_valid={long_valid}")
    
    # Try to convert to float if they're valid
    if lat_valid and long_valid:
        try:
            lat_float = float(latitude)
            long_float = float(longitude)
            
            st.success(f"Location detected: {lat_float:.6f}, {long_float:.6f}")
            
            # Save location to database
            st.write(f"DEBUG Step 4: Attempting database save...")
            st.write(f"  - userid: {userid}")
            st.write(f"  - timestamp: {timestamp}")
            st.write(f"  - lat: {lat_float}, long: {long_float}")
            
            try:
                with get_connection() as conn:
                    # Create new session
                    st.write("DEBUG: Executing session insert...")
                    cursor = conn.execute(LOC_UPDATE_SESSIONS_QUERY, {"userid": userid, "timestamp": timestamp})
                    session_id = cursor.lastrowid
                    st.write(f"DEBUG: Session created with ID: {session_id}")
                    
                    # Save GPS point
                    st.write("DEBUG: Executing point insert...")
                    conn.execute(
                        LOC_UPDATE_POINTS_QUERY, 
                        {"session_id": session_id, "timestamp": timestamp, "lat": lat_float, "long": long_float}
                    )
                    st.write("DEBUG: Point inserted")
                    
                    # Commit
                    st.write("DEBUG: Committing transaction...")
                    conn.commit()
                    st.write("DEBUG: Commit successful")
                    
                    st.success(f"Location saved to database! (Session ID: {session_id})")
            except Exception as e:
                st.error(f"Database error: {e}")
                st.exception(e)
        except (ValueError, TypeError) as e:
            st.error(f"Could not convert coordinates to numbers: {e}")
            st.write(f"Lat value: {repr(latitude)}, Long value: {repr(longitude)}")
    else:
        st.warning("Location data is None or zero")
        st.write(f"Lat: {repr(latitude)}, Long: {repr(longitude)}")
        st.info("""
        **Location not available.** This can happen because:
        - Browser hasn't received GPS signal yet (wait a few seconds and refresh)
        - Location services disabled on your device
        - Browser blocked location access
        - Running on HTTP instead of HTTPS (some browsers require secure connection)
        """)
else:
    st.warning("Waiting for location data from browser...")

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

# Debug: Show what we're querying
selected_date = date_scope[scope]
st.write(f"Searching for locations after: {selected_date}")

#download data
map_points = []

with get_connection() as conn:
    # Debug: Check total points in database
    total_points = conn.execute(
        "SELECT COUNT(*) FROM gps_points gp JOIN gps_sessions gs ON gp.session_id = gs.session_id WHERE gs.user_id = ?",
        (userid,)
    ).fetchone()[0]
    
    st.write(f"Total GPS points in database for your account: {total_points}")
    
    # Debug: Show all timestamps in database
    if total_points > 0:
        all_timestamps = conn.execute(
            "SELECT gp.recorded_at FROM gps_points gp JOIN gps_sessions gs ON gp.session_id = gs.session_id WHERE gs.user_id = ? ORDER BY gp.recorded_at DESC",
            (userid,)
        ).fetchall()
        st.write("Timestamps in database:", [t[0] for t in all_timestamps])
    
    # Get filtered points
    map_points = conn.execute(
        LOC_DOWNLOAD_QUERY, 
        {"userid": userid, "date_scope": selected_date.isoformat()}
    ).fetchall()
    
    st.write(f"Points matching {scope}: {len(map_points)}") 
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
