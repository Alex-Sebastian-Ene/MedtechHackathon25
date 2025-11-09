import streamlit as st
from databases.database import get_connection, initialize_database

initialize_database()

username = st.text_input("Enter your username")
password = st.text_input("Enter your password", type= "password")
nPassword = st.text_input("Confirm your password",type="password")

#query to update
SIGN_UP_QUERY = """
INSERT INTO users (username, password_hash, created_at) 
VALUES (:username, :password, 1);
"""

# Login button
hBtnContainer = st.container(horizontal=True)

if hBtnContainer.button("Sign Up"):
    if username == "" or password == "" or nPassword == "":
        st.error("Fields incomplete!")
    elif password != nPassword:
        st.error("Password don't match!")
    else:
        st.success("Sign up successful! Redirecting...")

        #upload information
        with get_connection() as conn:
            conn.execute(SIGN_UP_QUERY, {"username" : username, "password" : password})


        st.switch_page("pages/form.py")

if hBtnContainer.button("Back to Login",type="tertiary"):
    st.switch_page("login.py")