import streamlit as st
from streamlit_calendar import calendar
import pandas as pd
import numpy as np

# title
st.title("Depression Form")

questions_list=[
    {
        "title": "Today I have lost my appetite:",
        "type": "select_slider",
        "options": ["Not at all", "Slightly", "Moderately","Quite a bit","Completely"],
        "value": "Not at all",
    },
    {
        "title": "Today I feel like a failure:",
        "type": "select_slider",
        "options": ["Not at all", "Slightly", "Moderately","Quite a bit","Completely"],
        "value": "Not at all",
    },
    {
        "title": "Today I feel like I let myself down/I feel like I let my family down:",
        "type": "select_slider",
        "options": ["Not at all", "Slightly", "Moderately","Quite a bit","Completely"],
        "value": "Not at all",
    },
    {
        "title": "I want to hurt myself:",
        "type": "select_slider",
        "options": ["Not at all", "Once a week", "Every other day","Nearly every day","Every day"],
        "value": "Not at all",
    },
    {
        "title": "I feel like I am overeating:",
        "type": "select_slider",
        "options": ["Not at all", "Slightly", "Moderately","Quite a bit","Completely"],
        "value": "Not at all",
    },
    {
        "title": "How energetic do you feel?",
        "type": "feedback",
        "options": "faces",
    },
    {
        "title": "I am having trouble concentrating on things:",
        "type": "select_slider",
        "options": ["Not at all", "Slightly", "Moderately","Quite a bit","Completely"],
        "value": "Not at all",
    },
    {
        "title": "I have trouble sleeping:",
        "type": "select_slider",
        "options": ["Not at all", "Slightly", "Moderately","Quite a bit","Completely"],
        "value": "Not at all",
    },
    {
        "title": "I have little interest or pleasure in doing things:",
        "type": "select_slider",
        "options": ["Not at all", "Slightly", "Moderately","Quite a bit","Completely"],
        "value": "Not at all",
    },
    {
        "title": "I am moving or speaking so slowly that other people have noticed:",
        "type": "select_slider",
        "options": ["Not at all", "Slightly", "Moderately","Quite a bit","Completely"],
        "value": "Not at all",
    },
    {
        "title": "I am losing weight unintentionally",
        "type": "select_slider",
        "options": ["Not at all", "Slightly", "Moderately","Quite a bit","Completely"],
        "value": "Not at all",
    },
    {
        "title": "I am feeling agitated:",
        "type": "select_slider",
        "options": ["Not at all", "Slightly", "Moderately","Quite a bit","Completely"],
        "value": "Not at all",
    },
    {
        "title":"I am feeling these moods:",
        "type":"pills",
        "options":["Happy","Hopeful","Supported","Valued","Calm"],
    },
    {
        "title":"I am feeling these moods:",
        "type":"pills",
        "options":["Sad","Hopeless","Helpless","Worthless","Anxious"],
    }
]

# form containing each question, including text area
form=st.form(key="depressionForm")

for i,q in enumerate(questions_list):
    # creates a container for the question and input
    c = form.container(
        border=True
    )

    # creates subheader with question
    c.subheader(q["title"])

    # creates the particular type of input element
    match q["type"]:
        case "select_slider":
            c.select_slider(
                "",
                key="option "+str(i),
                options=q["options"],
                label_visibility="collapsed",
                value=q["value"],
            )
        case "feedback":
            c.feedback(
                key="option "+str(i),
                options=q["options"],
                default=2,
            )
        case "pills":
            c.pills(
                "",
                key="option "+str(i),
                options=q["options"],
                selection_mode="multi",
                label_visibility="collapsed",
            )

# text area element
text_container = form.container(border=True)
text_container.subheader("how are you feeling today?")
text_container.text_area(
    "",
    key="option -1",
    height="content",
    label_visibility="collapsed"
)

st.write(st.session_state)

# form submission button
submit = form.form_submit_button(
    label="Submit"
)

if submit:
    # gets relevant values
    object_to_send={}
    for i in range(-1,len(questions_list)):
        object_to_send["option "+str(i)]=st.session_state["option "+str(i)]
    
    # send object to database

    # return to home page
    st.switch_page("pages/homePage.py")
