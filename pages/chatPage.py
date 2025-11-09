import streamlit as st
from streamlit_calendar import calendar
import pandas as pd
import numpy as np
from medtech_hackathon25.ml.ollama_client import ChatSession, analyze_mood
from databases.database import (
    create_chat_session,
    save_chat_message,
    get_chat_session_messages,
    get_user_chat_sessions,
    update_chat_session_mood,
    get_chat_history_for_ollama
)

# Check if user is logged in
if "user_id" not in st.session_state or st.session_state.user_id is None:
    st.warning("Please log in to use the chat.")
    if st.button("Go to Login"):
        st.switch_page("app.py")
    st.stop()

# Title and user info
st.title("AI Health Assistant")
st.caption(f"👤 Logged in as: {st.session_state.get('username', 'Unknown')}")

# Initialize chat session
if "chat_session_id" not in st.session_state:
    st.session_state.chat_session_id = None
if "ollama_session" not in st.session_state:
    st.session_state.ollama_session = None

# Sidebar: Chat history
with st.sidebar:
    st.subheader("Chat History")
    
    if st.button("+ New Chat", type="primary"):
        st.session_state.chat_session_id = None
        st.session_state.ollama_session = None
        st.rerun()
    
    # Load previous chats
    previous_chats = get_user_chat_sessions(st.session_state.user_id, limit=10)
    for chat in previous_chats:
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button(f"💬 {chat['title'][:20]}...", key=f"load_{chat['session_id']}"):
                st.session_state.chat_session_id = chat['session_id']
                # Load history for Ollama
                history = get_chat_history_for_ollama(chat['session_id'])
                st.session_state.ollama_session = ChatSession(
                    system_prompt="You are a compassionate mental health assistant. Help users understand their emotions and provide supportive guidance.",
                    history=history
                )
                st.rerun()
        with col2:
            if chat['mood_score']:
                st.caption(f"😊 {chat['mood_score']}/10")

st.divider()

# Create new chat session if needed
if st.session_state.chat_session_id is None:
    session_id = create_chat_session(st.session_state.user_id, title="New Chat")
    st.session_state.chat_session_id = session_id
    st.session_state.ollama_session = ChatSession(
        system_prompt="You are a compassionate mental health assistant. Help users understand their emotions and provide supportive guidance."
    )
    # Save system message
    save_chat_message(session_id, "system", "You are a compassionate mental health assistant. Help users understand their emotions and provide supportive guidance.")

# Load messages from database
messages = get_chat_session_messages(st.session_state.chat_session_id)

# Display chat messages (skip system messages in UI)
for msg in messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# Chat input
prompt = st.chat_input("Ask me anything about your mental health...")
if prompt:
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Save user message to database
    save_chat_message(st.session_state.chat_session_id, "user", prompt)
    
    # Get response from Ollama
    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.ollama_session.ask(prompt)
                st.write(response)
        
        # Save assistant response to database
        save_chat_message(st.session_state.chat_session_id, "assistant", response)
        
        # Analyze mood using LLM and update session
        with st.spinner("Analyzing mood..."):
            history = st.session_state.ollama_session.history
            mood_score = analyze_mood(history, model=st.session_state.ollama_session.model)
            update_chat_session_mood(st.session_state.chat_session_id, mood_score)
        
        # Display mood score with interpretation
        if mood_score <= 2:
            mood_color = "🔴"
            mood_text = "Crisis - Please seek immediate help"
        elif mood_score <= 4:
            mood_color = "🟠"
            mood_text = "Very Low - Consider professional support"
        elif mood_score <= 6:
            mood_color = "🟡"
            mood_text = "Low - Managing but struggling"
        elif mood_score <= 8:
            mood_color = "🟢"
            mood_text = "Good - Coping well"
        else:
            mood_color = "🌟"
            mood_text = "Excellent - Thriving"
        
        st.info(f"{mood_color} Mood Score: {mood_score}/10 - {mood_text}")
        
        st.rerun()
        
    except Exception as e:
        st.error(f"Error communicating with AI: {str(e)}")
        st.info("Make sure Ollama is running. Run `ollama serve` in your terminal.")

st.divider()

# calendar

timeEvents=[
    {
        # "allDay":True,
        "title":"Event 1",
        "start":"2025-11-03",
        "end":"2025-11-03",
        "backgroundColor":"#FF6C6C",
        "borderColor":"#FF6C6C",
        "resourceId":"a",
    },
    {
        "allDay":True,
        "title":"Event 3",
        "start":"2025-11-03",
        "end":"2025-11-03",
        "backgroundColor":"#FF6C6C",
        "borderColor":"#FF6C6C",
        "resourceId":"a",
    },
]

def exampleFunc():
    st.rerun()

st.set_page_config(page_title="Demo for streamlit-calendar", page_icon="📆")

events = [
    {
        "title": "Event 1",
        "color": "#FF6C6C",
        "start": "2023-07-03",
        "end": "2023-07-05",
        "resourceId": 1,
    },
    {
        "title": "Event 2",
        "color": "#FFBD45",
        "start": "2023-07-01",
        "end": "2023-07-10",
        "resourceId": 2,
    },
]

calendar_resources = [
    {"id": 1, "building": "Building A", "title": "Room A"},
    {"id": 2, "building": "Building A", "title": "Room B"},
]

calendar_options = {
    "editable": "true",
    "navLinks": "true",
    "resources": calendar_resources,
    "selectable": "true",
    "headerToolbar": {
        "left": "today prev,next",
        "center": "title",
        "right": "dayGridDay,dayGridWeek,dayGridMonth",
    },
    "initialDate": "2023-07-01",
    "initialView": "dayGridMonth",
}

call_event=[
    'eventClick'
]

state = calendar(
    events=events,
    options=calendar_options,
    callbacks=call_event,
    custom_css="""
    .fc-event-past {
        opacity: 0.8;
    }
    .fc-event-time {
        font-style: italic;
    }
    .fc-event-title {
        font-weight: 700;
    }
    .fc-toolbar-title {
        font-size: 2rem;
    }
    """,
    key="timegrid",
)

beenClicked=False
# Handle calendar interactions
if state.get("eventClick"):
    # Get clicked event details
    clicked_event = state["eventClick"]["event"]
    event_title = clicked_event["title"]
    event_start = clicked_event["start"]

    st.session_state.current_event=clicked_event
    st.switch_page("pages/formView.py")

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

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)


st.subheader('Number of pickups by hour')
hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
st.bar_chart(hist_values)
st.line_chart(hist_values)

hour_to_filter = st.slider('hour',0,23,17)

filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
st.subheader(f'Map of all pickups at {hour_to_filter}:00')
st.map(filtered_data)

enable = st.checkbox("Enable camera")
picture = st.camera_input("Take a picture", disabled=not enable)

# if picture:
#     st.image(picture)
