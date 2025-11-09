import streamlit as st
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
        st.switch_page("login.py")
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
