import streamlit as st
from streamlit_calendar import calendar
import pandas as pd
import numpy as np

# title
st.title("Example Title")

# chatbot page elements
prompt = st.chat_input("Say something")
if prompt:
    newmessage = st.chat_message("user")
    newmessage.write(prompt)

messages=[
    {"sender":"assistant","text":"hello!"},
    {"sender":"user","text":"response"},
]

for m in messages:
    message = st.chat_message(m["sender"])
    message.write(m["text"])

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