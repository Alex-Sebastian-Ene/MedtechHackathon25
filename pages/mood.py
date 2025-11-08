import streamlit as st
import pandas as pd
import plotly.express as px
from databases.database import get_user_mood_history
from datetime import datetime

# Check if user is logged in
if "user_id" not in st.session_state or st.session_state.user_id is None:
    st.warning("Please log in to view your mood history.")
    if st.button("Go to Login"):
        st.switch_page("app.py")
    st.stop()

st.title("Mood History")
st.caption(f"👤 Logged in as: {st.session_state.get('username', 'Unknown')}")

# Fetch mood data from database
mood_data = get_user_mood_history(st.session_state.user_id, limit=100)

if not mood_data:
    st.info("No mood entries yet. Complete an emotion scan to start tracking your mood!")
else:
    # Convert to DataFrame
    df = pd.DataFrame(mood_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Display summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Entries", len(df))
    with col2:
        avg_mood = df['mood_score'].mean()
        st.metric("Average Mood", f"{avg_mood:.1f}/10")
    with col3:
        latest_mood = df.iloc[0]['mood_score']
        st.metric("Latest Mood", f"{latest_mood}/10")
    
    st.divider()
    
    # Plot mood over time
    fig = px.line(
        df.sort_values('timestamp'),
        x='timestamp',
        y='mood_score',
        title='Mood Score Over Time',
        markers=True
    )
    fig.update_layout(
        yaxis_range=[0, 10],
        xaxis_title="Date & Time",
        yaxis_title="Mood Score"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Display recent entries
    st.subheader("Recent Entries")
    for entry in mood_data[:10]:
        with st.expander(f"{entry['timestamp']} - Mood: {entry['mood_score']}/10"):
            st.write(f"**Score:** {entry['mood_score']}/10")
            st.write(f"**Notes:** {entry['notes']}")



