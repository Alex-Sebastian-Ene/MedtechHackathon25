import streamlit as st
import pandas as pd
import plotly.express as px
from databases.database import get_user_mood_history
from datetime import datetime, date, timedelta

# Check if user is logged in
if "user_id" not in st.session_state or st.session_state.user_id is None:
    st.warning("Please log in to view your mood history.")
    if st.button("Go to Login"):
        st.switch_page("login.py")
    st.stop()

st.title("Mood History")
st.caption(f"👤 Logged in as: {st.session_state.get('username', 'Unknown')}")

# Date scopes
# Last 24 hours
today = datetime.combine(date.today(), datetime.min.time())

# This week
now = datetime.now()
week = now - timedelta(days=now.weekday())  # weekday() → Monday=0, Sunday=6
week = week.replace(hour=0, minute=0, second=0, microsecond=0)

# Last month
month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

# Scope dictionary
date_scope = {"Past 24hrs": today, "Past Week": week, "Past Month": month}

# User gets to choose scope
scope = st.select_slider(
    "Select a time period",
    options=[
        "Past 24hrs",
        "Past Week",
        "Past Month",
    ]
)

# Get the selected date threshold
selected_date = date_scope[scope]

# Fetch mood data from database
mood_data = get_user_mood_history(st.session_state.user_id, limit=1000)

if not mood_data:
    st.info("No mood entries yet. Complete an emotion scan or depression assessment to start tracking your mood!")
else:
    # Convert to DataFrame
    df = pd.DataFrame(mood_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Filter by selected date scope
    df = df[df['timestamp'] >= selected_date]
    
    # Check if filtered data is empty
    if len(df) == 0:
        st.warning(f"No mood entries found for {scope}. Try selecting a different time period.")
        st.stop()
    
    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Entries", len(df))
    with col2:
        avg_mood = df['mood_score'].mean()
        st.metric("Average Mood", f"{avg_mood:.1f}/10")
    with col3:
        latest_mood = df.iloc[0]['mood_score']
        st.metric("Latest Mood", f"{latest_mood}/10")
    with col4:
        if len(df) > 1:
            trend = df.iloc[0]['mood_score'] - df.iloc[-1]['mood_score']
            st.metric("Trend", f"{trend:+.1f}", delta=f"{trend:+.1f}")
        else:
            st.metric("Trend", "N/A")
    
    st.divider()
    
    # Plot mood over time
    fig = px.line(
        df.sort_values('timestamp'),
        x='timestamp',
        y='mood_score',
        title=f'Mood Score Over Time - {scope}',
        markers=True
    )
    fig.update_layout(
        yaxis_range=[0, 10],
        xaxis_title="Date & Time",
        yaxis_title="Mood Score"
    )
    fig.update_traces(
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8, color='#ff7f0e')
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Display recent entries
    st.subheader(f"Entries for {scope}")
    
    # Show all entries in the filtered period (up to 50)
    display_count = min(len(df), 50)
    for i in range(display_count):
        entry = df.iloc[i]
        
        # Color code based on mood score
        if entry['mood_score'] >= 7:
            emoji = "😊"
            color = "green"
        elif entry['mood_score'] >= 4:
            emoji = "😐"
            color = "orange"
        else:
            emoji = "😔"
            color = "red"
        
        with st.expander(f"{emoji} {entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} - Mood: {entry['mood_score']}/10"):
            col_a, col_b = st.columns([1, 3])
            with col_a:
                st.metric("Score", f"{entry['mood_score']}/10")
            with col_b:
                if entry.get('notes'):
                    st.write("**Notes:**")
                    st.write(entry['notes'])
                else:
                    st.caption("No notes recorded")
    
    if len(df) > 50:
        st.info(f"Showing 50 of {len(df)} entries. Adjust time period to see more recent data.")



