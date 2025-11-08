import streamlit as st
from streamlit_calendar import calendar
import pandas as pd
import numpy as np

# title
st.title("Chatbot Page")

# messages to pre-load (would be fetched from database)
messages=[
    {
        "sender":"user",
        "text":"hello! generic query"
    },
    {
        "sender":"assistant",
        "text":"response"
    },
    {
        "sender":"user",
        "text":"hello! generic query"
    },
    {
        "sender":"assistant",
        "text":"response"
    },
    {
        "sender":"user",
        "text":"hello! generic query"
    },
    {
        "sender":"assistant",
        "text":"response"
    },
    {
        "sender":"user",
        "text":"hello! generic query"
    },
    {
        "sender":"assistant",
        "text":"response"
    },
    {
        "sender":"user",
        "text":"hello! generic query"
    },
    {
        "sender":"assistant",
        "text":"response"
    },
]

# button to go back to home page

btnContainer = st.container(horizontal=True,horizontal_alignment='distribute')

returnButton = btnContainer.button(
    "Return to Home",
    icon=":material/arrow_back_ios:",
)

with btnContainer.popover("Additional Resources",icon=":material/phone:"):
    st.markdown("resource 1")
    st.markdown("resource 1")
    st.markdown("resource 1")


if returnButton:
    st.switch_page("pages/homePage.py")

# chat input element
prompt = st.chat_input("Say something")
if prompt:
    # newmessage = st.chat_message("user")
    # newmessage.write(prompt)

    # replace this with a push to the database,
    # and add the code that triggers LLM to create a response
    messages.append({
        "sender":"user",
        "text":prompt,
    })

# streamlit recommends avoiding scrolling containers of >500px for mobile phone compatibility reasons
chatContainer = st.container(height=800)

# generates element for each message
for m in messages:
    message = chatContainer.chat_message(m["sender"])
    message.write(m["text"])