import streamlit as st
from databases.database import get_connection, initialize_database

initialize_database()

username = st.text_input("Enter a username")
password = st.text_input("Enter a password", type= "password")
email = st.text_input("Enter your email")

#query to update
SIGN_UP_QUERY = """
INSERT INTO users (username, password_hash, email, created_at) 
VALUES (:username, :password, :email, 1);
"""

# Login button
if st.button("Sign Up"):
    if username != "" and password != "" and email != "":
        st.success("sign up successful! Redirecting...")

        #upload information
        with get_connection() as conn:
            conn.execute(SIGN_UP_QUERY, {"username" : username, "password" : password, "email" : email })


        st.switch_page("app.py")
    else:
        st.error("Fields Incomplete")
