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
timestamp = datetime.now().isoformat()

# Initialize location tracking flag in session state
if 'location_saved_this_session' not in st.session_state:
    st.session_state.location_saved_this_session = False

# Automatic location tracking - runs on page load
if not st.session_state.location_saved_this_session:
    try:
        current_loc = geo()
        
        if current_loc:
            auto_lat = current_loc.get('latitude')
            auto_lon = current_loc.get('longitude')
            
            # Check if valid coordinates
            if auto_lat and auto_lon and auto_lat != 0 and auto_lon != 0:
                # Automatically save to database
                try:
                    with get_connection() as conn:
                        cursor = conn.execute(LOC_UPDATE_SESSIONS_QUERY, {"userid": userid, "timestamp": timestamp})
                        session_id = cursor.lastrowid
                        
                        conn.execute(
                            LOC_UPDATE_POINTS_QUERY, 
                            {"session_id": session_id, "timestamp": timestamp, "lat": auto_lat, "long": auto_lon}
                        )
                        
                        conn.commit()
                        st.session_state.location_saved_this_session = True
                        st.success(f"Location automatically saved: {auto_lat:.6f}, {auto_lon:.6f}")
                except Exception as e:
                    st.warning(f"Could not auto-save location: {e}")
            else:
                st.info("Location tracking enabled but no GPS signal detected. Your location will be tracked when available.")
    except Exception as e:
        st.info("Automatic location tracking is active. Allow location access if prompted by your browser.")

# Show tracking status
with st.expander("Location Tracking Status"):
    if st.session_state.location_saved_this_session:
        st.success("✅ Location tracked this session")
    else:
        st.warning("⏳ Waiting for location signal...")
    
    if st.button("Force Refresh Location"):
        st.session_state.location_saved_this_session = False
        st.rerun()

st.divider()

# Date scopes
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

selected_date = date_scope[scope]

# Download data
map_points = []

with get_connection() as conn:
    # Check total points in database
    total_points = conn.execute(
        "SELECT COUNT(*) FROM gps_points gp JOIN gps_sessions gs ON gp.session_id = gs.session_id WHERE gs.user_id = ?",
        (userid,)
    ).fetchone()[0]
    
    if total_points > 0:
        st.info(f"Found {total_points} GPS points in your history")
        
        # Get filtered points
        map_points = conn.execute(
            LOC_DOWNLOAD_QUERY, 
            {"userid": userid, "date_scope": selected_date.isoformat()}
        ).fetchall()
        
        if map_points:
            # Convert to DataFrame
            df = pd.DataFrame(map_points, columns=['lon', 'lat', 'recorded_at'])
            
            st.success(f"Showing {len(df)} location(s) for {scope}")
            
            # Create map
            marker_color = [100, 149, 237]  # Soft blue
            
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
                zoom=12,
                pitch=0
            )
            
            r = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                map_style='mapbox://styles/mapbox/light-v10',
            )
            
            st.pydeck_chart(r)
            
            # Show data table
            with st.expander("View Location History"):
                st.dataframe(df[['lat', 'lon', 'recorded_at']])
        else:
            st.info(f"No locations found for {scope}. Try a different time range.")
    else:
        st.warning("No GPS data found. Use the form above to add locations.")
        st.info("""
        **To add your first location:**
        1. Get your coordinates from Google Maps (right-click on map → click coordinates)
        2. Enter latitude and longitude in the fields above
        3. Click "Save This Location"
        """)
