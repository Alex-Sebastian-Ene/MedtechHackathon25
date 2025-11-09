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

# Initialize tracking variables in session state
if 'last_location_save' not in st.session_state:
    st.session_state.last_location_save = None
if 'location_count' not in st.session_state:
    st.session_state.location_count = 0

# Automatic location tracking - tries to get location
try:
    current_loc = geo()
    
    if current_loc:
        auto_lat = current_loc.get('latitude')
        auto_lon = current_loc.get('longitude')
        
        # Check if valid coordinates
        if auto_lat and auto_lon and auto_lat != 0 and auto_lon != 0:
            # Check if enough time has passed (60 seconds) or first save
            current_time = datetime.now()
            should_save = False
            
            if st.session_state.last_location_save is None:
                should_save = True
            else:
                time_diff = (current_time - st.session_state.last_location_save).total_seconds()
                if time_diff >= 60:  # 60 seconds = 1 minute
                    should_save = True
            
            if should_save:
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
                        st.session_state.last_location_save = current_time
                        st.session_state.location_count += 1
                        st.success(f"Location #{st.session_state.location_count} saved: {auto_lat:.6f}, {auto_lon:.6f}")
                except Exception as e:
                    st.warning(f"Could not auto-save location: {e}")
            else:
                # Show countdown to next save
                time_since_last = (current_time - st.session_state.last_location_save).total_seconds()
                seconds_remaining = int(60 - time_since_last)
                st.info(f"Location detected: {auto_lat:.6f}, {auto_lon:.6f} - Next save in {seconds_remaining}s")
        else:
            st.info("Location tracking enabled but no GPS signal detected.")
except Exception as e:
    st.info("Automatic location tracking is active. Allow location access if prompted by your browser.")

st.divider()

# View mode selection
view_mode = st.radio(
    "Map View:",
    ["Simple Map", "Local View", "World View"],
    horizontal=True,
    help="Simple Map: Basic view | Local View: Zoomed with paths | World View: Global overview"
)

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
            # Convert to DataFrame and sort by time
            df = pd.DataFrame(map_points, columns=['lon', 'lat', 'recorded_at'])
            df = df.sort_values('recorded_at')  # Sort chronologically
            
            st.success(f"Showing {len(df)} location(s) for {scope}")
            
            # Simple Map Mode - use PyDeck but simplified
            if view_mode == "Simple Map":
                # Create simple layers
                layers = []
                
                # Path connecting points
                if len(df) > 1:
                    path_data = [{
                        'path': df[['lon', 'lat']].values.tolist(),
                        'color': [100, 149, 237]  # Blue line
                    }]
                    path_layer = pdk.Layer(
                        "PathLayer",
                        data=path_data,
                        get_path='path',
                        get_color='color',
                        width_min_pixels=2,
                        pickable=False
                    )
                    layers.append(path_layer)
                
                # Scatter points
                scatter_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=df,
                    get_position='[lon, lat]',
                    get_color=[255, 80, 80],  # Red
                    get_radius=500,
                    pickable=True,
                    auto_highlight=True
                )
                layers.append(scatter_layer)
                
                # Auto-zoom to fit all points
                view_state = pdk.ViewState(
                    longitude=df['lon'].mean(),
                    latitude=df['lat'].mean(),
                    zoom=10,
                    pitch=0
                )
                
                r = pdk.Deck(
                    layers=layers,
                    initial_view_state=view_state,
                    map_style='mapbox://styles/mapbox/streets-v11',
                    tooltip={"text": "Location: {lat}, {lon}\n{recorded_at}"}
                )
                
                st.pydeck_chart(r)
                
                # Show path summary
                if len(df) > 1:
                    st.info(f"🔵 Path connects {len(df)} locations chronologically")
                
                # Show data table
                with st.expander("View Location History"):
                    df_display = df.copy()
                    df_display.insert(0, 'Order', range(1, len(df_display) + 1))
                    st.dataframe(df_display[['Order', 'recorded_at', 'lat', 'lon']])
            
            else:
                # Advanced views with PyDeck
                # Prepare path data - connect points in chronological order
                path_data = [{
                    'path': df[['lon', 'lat']].values.tolist(),
                    'color': [100, 149, 237, 200]  # RGBA - blue with transparency
                }]
                
                # Create layers
                layers = []
                
                # Layer 1: Path line connecting all points
                if len(df) > 1:
                    path_layer = pdk.Layer(
                        "PathLayer",
                        data=path_data,
                        get_path='path',
                        get_color='color',
                        width_min_pixels=3,
                        pickable=False
                    )
                    layers.append(path_layer)
                
                # Layer 2: Scatter points for each location
                if view_mode == "World View":
                    # Larger markers for world view
                    scatter_layer = pdk.Layer(
                        "ScatterplotLayer",
                        data=df,
                        get_position='[lon, lat]',
                        get_color=[255, 100, 100, 200],  # Red markers
                        get_radius=100000,  # 100km radius for world view
                        pickable=True,
                        auto_highlight=True
                    )
                else:
                    # Normal markers for local view
                    scatter_layer = pdk.Layer(
                        "ScatterplotLayer",
                        data=df,
                        get_position='[lon, lat]',
                        get_color=[255, 100, 100, 200],  # Red markers
                        get_radius=300,
                        pickable=True,
                        auto_highlight=True
                    )
                layers.append(scatter_layer)
                
                # Layer 3: Larger marker for most recent location
                latest_point = df.iloc[-1:].copy()
                if view_mode == "World View":
                    latest_radius = 150000  # 150km for world view
                else:
                    latest_radius = 500  # 500m for local view
                    
                latest_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=latest_point,
                    get_position='[lon, lat]',
                    get_color=[50, 200, 50, 255],  # Green for latest
                    get_radius=latest_radius,
                    pickable=True
                )
                layers.append(latest_layer)
                
                # Set view based on mode
                if view_mode == "World View":
                    view_state = pdk.ViewState(
                        longitude=0,  # Center of world
                        latitude=20,
                        zoom=1.5,  # World zoom level
                        pitch=0
                    )
                else:
                    view_state = pdk.ViewState(
                        longitude=df['lon'].mean(),
                        latitude=df['lat'].mean(),
                        zoom=12,  # Local zoom level
                        pitch=0
                    )
                
                # Choose map style based on view mode
                if view_mode == "World View":
                    map_style = 'mapbox://styles/mapbox/streets-v12'  # Detailed with country borders
                else:
                    map_style = 'mapbox://styles/mapbox/light-v10'  # Clean local view
                
                r = pdk.Deck(
                    layers=layers,
                    initial_view_state=view_state,
                    map_style=map_style,
                    tooltip={
                        "text": "Location tracked at:\n{recorded_at}"
                    }
                )
                
                st.pydeck_chart(r)
                
                # Show coordinates for verification
                if view_mode == "World View":
                    st.info(f"📍 Location spread: {df['lat'].min():.2f}° to {df['lat'].max():.2f}° latitude, {df['lon'].min():.2f}° to {df['lon'].max():.2f}° longitude")
                    
                    # Identify approximate locations
                    with st.expander("📍 Location Details"):
                        for idx, row in df.iterrows():
                            lat, lon = row['lat'], row['lon']
                            timestamp = row['recorded_at']
                            
                            # Simple hemisphere identification
                            lat_hemisphere = "N" if lat >= 0 else "S"
                            lon_hemisphere = "E" if lon >= 0 else "W"
                            
                            st.write(f"**Point {idx+1}:** {abs(lat):.4f}°{lat_hemisphere}, {abs(lon):.4f}°{lon_hemisphere} - {timestamp}")
                            
                            # Give rough location hints
                            if -180 <= lon < -30:
                                region = "Americas"
                            elif -30 <= lon < 60:
                                region = "Europe/Africa"
                            elif 60 <= lon < 150:
                                region = "Asia"
                            else:
                                region = "Pacific/Oceania"
                            
                            st.caption(f"   Approximate region: {region}")
                
                # Legend
                st.markdown("""
                **Map Legend:**
                - 🔴 Red dots: Location points
                - 🟢 Green dot: Most recent location
                - 🔵 Blue line: Path connecting locations in chronological order
                """)
                
                # Show data table
                with st.expander("View Location History"):
                    # Add index to show order
                    df_display = df.copy()
                    df_display.insert(0, 'Order', range(1, len(df_display) + 1))
                    st.dataframe(df_display[['Order', 'recorded_at', 'lat', 'lon']])
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

# Auto-refresh mechanism at the very end (after all content is displayed)
import time

# Show refresh status in sidebar
with st.sidebar:
    st.divider()
    st.subheader("🔄 Auto-Tracking")
    if st.session_state.location_count > 0:
        st.success(f"✅ {st.session_state.location_count} points saved")
        if st.session_state.last_location_save:
            time_since = (datetime.now() - st.session_state.last_location_save).total_seconds()
            if time_since < 60:
                st.info(f"⏱️ Next save in {int(60 - time_since)}s")
    else:
        st.warning("⏳ Waiting for GPS...")
    
    st.caption("Auto-refresh: Every 15 seconds")
    
    if st.button("🔄 Refresh Now"):
        st.rerun()

# Wait 15 seconds then automatically refresh the page
time.sleep(15)
st.rerun()
