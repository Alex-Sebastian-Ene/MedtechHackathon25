import sqlite3
import streamlit as st
from databases.database import get_connection, initialize_database
from datetime import datetime

# Initialize database
initialize_database()

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "username" not in st.session_state:
    st.session_state.username = None

st.title("Login Page")

# Login inputs
username = st.text_input("Username", placeholder=" John Smith")
password = st.text_input("Password", type="password")


USER_RETRIEVE = """
SELECT
	user_id
FROM users
WHERE username = :username
AND 
password_hash = :password;
"""
with get_connection() as conn:
    data = conn.execute(USER_RETRIEVE, { "username": username, "password": password}).fetchall()

hBtnContainer=st.container(horizontal=True)


# Login button
if hBtnContainer.button("Login"):
    if username and password:
        with get_connection() as conn:
            # Check if user exists
            user = conn.execute(
                "SELECT user_id, username FROM users WHERE username = ? AND password_hash = ?",
                (username, password)
            ).fetchone()
            
            if user:
                st.session_state.logged_in = True
                st.session_state.user_id = user[0]
                st.session_state.username = user[1]
                st.success("Login successful! Redirecting...")
                st.switch_page("pages/home.py")
            else:
                # Try to create new user (simple auto-registration)
                try:
                    cursor = conn.execute(
                        "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                        (username, password, datetime.now().isoformat())
                    )
                    conn.commit()
                    st.session_state.logged_in = True
                    st.session_state.user_id = cursor.lastrowid
                    st.session_state.username = username
                    st.success("Account created! Redirecting...")
                    st.switch_page("pages/home.py")
                except sqlite3.IntegrityError:
                    st.error("Invalid username or password")
    else:
        st.error("Please enter both username and password")

if hBtnContainer.button("Don't have an account? Sign up!",type="tertiary"):
    st.switch_page("pages/signup.py")