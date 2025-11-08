import streamlit as st
from streamlit_calendar import calendar
import pandas as pd
import numpy as np

# title
st.title("Example Title")

questions_list=[
    {
        "title": "How depressed are you feeling?",
        "type": "select_slider",
        "options": [1,2,3,4,5,6,7,8,9,10],
    },
    {
        "title": "On a scale of frown to smile, how are you doing?",
        "type": "feedback",
        "options": "faces",
    },
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
                value=5,
            )
        case "feedback":
            c.feedback(
                key="option "+str(i),
                options=q["options"],
                default=2,
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
