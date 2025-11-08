import streamlit as st

st.title("Login Page")

# Login inputs
username = st.text_input("Username", placeholder=" John Smith")
password = st.text_input("Password", type="password")

# Login button
if st.button("Login"):
    if username == "admin" and password == "123":
        st.success("Login successful! Redirecting...")
        st.switch_page("pages/home.py")
    else:
        st.error("Invalid username or password")