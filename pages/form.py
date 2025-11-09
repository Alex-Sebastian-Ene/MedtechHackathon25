"""
Depression Assessment Form - PHQ-9 Based Mental Health Screening

Features:
- Saves all individual responses to questionnaire_responses table
- Calculates weighted average mood score (1-5 scale, lower = worse)
- Uses actual response count (not max possible) for denominator
- Multiple choice pills contribute to running average based on selection counts
- LLM analysis of text responses for comprehensive assessment
- Combined score (60% structured, 40% LLM) saved to mood_entries
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from databases.database import (
    create_questionnaire,
    add_questionnaire_question,
    save_questionnaire_response,
    get_connection
)
from medtech_hackathon25.ml.ollama_client import get_response

# Initialize session state
if "user_id" not in st.session_state:
    st.session_state.user_id = 1  # Default user for testing

if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False
    
if "redirect_page" not in st.session_state:
    st.session_state.redirect_page = None

# Handle redirects
if st.session_state.redirect_page:
    page = st.session_state.redirect_page
    st.session_state.redirect_page = None
    st.session_state.form_submitted = False
    st.switch_page(page)

# title
st.title("Mood and Anxiety Tracking Form")

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

# form submission button
submit = form.form_submit_button(
    label="Submit Assessment"
)

def calculate_mood_score_from_responses(responses: dict) -> float:
    """
    Calculate mood score from questionnaire responses.
    Lower score = worse mental health (1-5 scale)
    Uses weighted average (divide by actual count, not max possible).
    """
    total_weighted_score = 0
    total_weights = 0
    
    # Mapping for slider responses (depression indicators)
    severity_map = {
        "Not at all": 0,
        "Slightly": 1,
        "Moderately": 2,
        "Quite a bit": 3,
        "Completely": 4,
        "Once a week": 1,
        "Every other day": 2,
        "Nearly every day": 3,
        "Every day": 4
    }
    
    # Process each response
    for i in range(len(questions_list)):
        key = f"option {i}"
        if key not in responses:
            continue
            
        response = responses[key]
        question = questions_list[i]
        
        if question["type"] == "select_slider":
            # Higher severity = worse mood
            severity = severity_map.get(response, 0)
            
            # Weight critical questions more heavily
            weight = 1.0
            if "hurt myself" in question["title"].lower():
                weight = 2.5  # Self-harm is critical
            elif "failure" in question["title"].lower():
                weight = 1.5
            elif "let" in question["title"].lower() and "down" in question["title"].lower():
                weight = 1.5
            
            # Normalize severity to 0-1 scale, then apply weight
            normalized_severity = severity / 4.0  # 0-4 → 0-1
            total_weighted_score += normalized_severity * weight
            total_weights += weight
            
        elif question["type"] == "feedback":
            # Feedback scale: 0-4, invert it (lower energy = worse)
            energy_level = response if isinstance(response, int) else 2
            weight = 1.2
            
            # Normalize and invert (4-energy gives us inverse)
            normalized_score = (4 - energy_level) / 4.0  # 0-1 scale
            total_weighted_score += normalized_score * weight
            total_weights += weight
            
        elif question["type"] == "pills":
            if isinstance(response, list):
                # Positive moods decrease score, negative moods increase it
                positive_moods = ["Happy", "Hopeful", "Supported", "Valued", "Calm"]
                negative_moods = ["Sad", "Hopeless", "Helpless", "Worthless", "Anxious"]
                
                positive_count = sum(1 for mood in response if mood in positive_moods)
                negative_count = sum(1 for mood in response if mood in negative_moods)
                
                weight = 1.3
                
                # Calculate as ratio of negative to total possible
                # More negative = higher score (worse)
                total_selected = len(response)
                if total_selected > 0:
                    # Negative ratio increased by positive reduction
                    negative_ratio = negative_count / 5.0  # Max 5 negative moods
                    positive_ratio = positive_count / 5.0  # Max 5 positive moods
                    
                    # Combined score: negative adds, positive subtracts
                    pill_score = negative_ratio - (positive_ratio * 0.5)  # Positive helps less
                    pill_score = max(0, min(1, pill_score))  # Clamp to 0-1
                    
                    total_weighted_score += pill_score * weight
                    total_weights += weight
    
    if total_weights == 0:
        return 5.0  # Default neutral
    
    # Calculate weighted average (0-1 scale)
    average_score = total_weighted_score / total_weights
    
    # Convert to 1-5 scale (inverted: lower = worse)
    # average_score of 1.0 (worst) → mood_score of 1.0
    # average_score of 0.0 (best) → mood_score of 5.0
    mood_score = 5.0 - (average_score * 4.0)
    
    # Clamp to 1-5 range
    mood_score = max(1.0, min(5.0, mood_score))
    
    return round(mood_score, 2)


def analyze_text_with_llm(text_response: str) -> tuple[float, str]:
    """
    Send text response to LLM for mood analysis.
    Returns (score, explanation) where score is 1-5 (lower = worse).
    """
    if not text_response or text_response.strip() == "":
        return 3.0, "No text response provided."
    
    prompt = f"""You are a mental health assessment tool. Analyze the following text response from a depression screening questionnaire.

User's response: "{text_response}"

Provide a mental health score from 1-5:
- 1: Severe depression, crisis indicators, immediate concern
- 2: Strong depression, significant distress
- 3: Moderate concern, some negative indicators
- 4: Mild concern, mostly coping
- 5: Positive mental health, good coping

Respond in this exact format:
SCORE: [number]
REASON: [brief explanation]"""

    try:
        llm_response = get_response(prompt, system_prompt="You are a clinical mental health assessment assistant.")
        
        # Parse response
        score_line = [line for line in llm_response.split('\n') if 'SCORE:' in line.upper()]
        reason_line = [line for line in llm_response.split('\n') if 'REASON:' in line.upper()]
        
        if score_line:
            score_text = score_line[0].split(':')[1].strip()
            import re
            numbers = re.findall(r'\b([1-5])\b', score_text)
            llm_score = float(numbers[0]) if numbers else 3.0
        else:
            llm_score = 3.0
            
        if reason_line:
            reason = reason_line[0].split(':', 1)[1].strip()
        else:
            reason = "Analysis completed."
            
        return llm_score, reason
        
    except Exception as e:
        st.warning(f"LLM analysis unavailable: {e}")
        return 3.0, "LLM analysis failed, using default score."


if submit:
    with st.spinner("Processing your assessment..."):
        # Collect all responses
        responses = {}
        for i in range(-1, len(questions_list)):
            key = f"option {i}"
            if key in st.session_state:
                responses[key] = st.session_state[key]
        
        # Calculate mood score from structured responses
        structured_score = calculate_mood_score_from_responses(responses)
        
        # Analyze text response with LLM
        text_response = responses.get("option -1", "")
        llm_score, llm_explanation = analyze_text_with_llm(text_response)
        
        # Combined final score (weighted average)
        # Structured responses: 60%, LLM analysis: 40%
        final_score = (structured_score * 0.6) + (llm_score * 0.4)
        final_score = round(final_score, 2)
        
        # Save to database
        try:
            with get_connection() as conn:
                # First, create or get the questionnaire
                questionnaire_id = None
                existing_q = conn.execute(
                    "SELECT questionnaire_id FROM questionnaires WHERE title = ? AND is_active = 1",
                    ("Depression Assessment Form",)
                ).fetchone()
                
                if existing_q:
                    questionnaire_id = existing_q[0]
                else:
                    # Create questionnaire
                    cursor = conn.execute(
                        "INSERT INTO questionnaires (title, description, created_at, is_active) VALUES (?, ?, ?, ?)",
                        ("Depression Assessment Form", "PHQ-9 based depression screening", datetime.now().isoformat(), 1)
                    )
                    questionnaire_id = cursor.lastrowid
                    
                    # Add questions to questionnaire
                    for idx, question in enumerate(questions_list):
                        # Map UI question types to database allowed types
                        question_type_map = {
                            "select_slider": "scale",
                            "feedback": "scale",
                            "pills": "multiple_choice"
                        }
                        db_question_type = question_type_map.get(question["type"], question["type"])
                        
                        conn.execute(
                            """
                            INSERT INTO questionnaire_questions 
                            (questionnaire_id, question_text, question_type, options, order_index, is_required)
                            VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (
                                questionnaire_id,
                                question["title"],
                                db_question_type,
                                str(question.get("options", [])),
                                idx,
                                1
                            )
                        )
                    
                    # Add text question
                    conn.execute(
                        """
                        INSERT INTO questionnaire_questions 
                        (questionnaire_id, question_text, question_type, options, order_index, is_required)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (questionnaire_id, "How are you feeling today?", "text", None, len(questions_list), 0)
                    )
                    conn.commit()
                
                # Get question IDs
                question_rows = conn.execute(
                    """
                    SELECT question_id, order_index FROM questionnaire_questions
                    WHERE questionnaire_id = ?
                    ORDER BY order_index
                    """,
                    (questionnaire_id,)
                ).fetchall()
                
                # Save individual responses
                severity_map = {
                    "Not at all": 0, "Slightly": 1, "Moderately": 2, "Quite a bit": 3, "Completely": 4,
                    "Once a week": 1, "Every other day": 2, "Nearly every day": 3, "Every day": 4
                }
                
                for question_id, order_idx in question_rows:
                    if order_idx < len(questions_list):
                        # Regular question
                        key = f"option {order_idx}"
                        if key in responses:
                            response_value = responses[key]
                            
                            # Convert to numeric value where applicable
                            if isinstance(response_value, str):
                                numeric_value = severity_map.get(response_value, None)
                                text_value = response_value
                            elif isinstance(response_value, int):
                                numeric_value = float(response_value)
                                text_value = str(response_value)
                            elif isinstance(response_value, list):
                                # Multiple choice - save as comma-separated
                                text_value = ", ".join(response_value)
                                numeric_value = float(len(response_value))
                            else:
                                text_value = str(response_value)
                                numeric_value = None
                            
                            conn.execute(
                                """
                                INSERT INTO questionnaire_responses
                                (user_id, questionnaire_id, question_id, response_text, response_value, answered_at)
                                VALUES (?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    st.session_state.user_id,
                                    questionnaire_id,
                                    question_id,
                                    text_value,
                                    numeric_value,
                                    datetime.now().isoformat()
                                )
                            )
                    else:
                        # Text response question
                        if text_response:
                            conn.execute(
                                """
                                INSERT INTO questionnaire_responses
                                (user_id, questionnaire_id, question_id, response_text, response_value, answered_at)
                                VALUES (?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    st.session_state.user_id,
                                    questionnaire_id,
                                    question_id,
                                    text_response,
                                    None,
                                    datetime.now().isoformat()
                                )
                            )
                
                # Convert final score to 1-10 scale for mood_entries table
                mood_score_10 = int(round(final_score * 2))  # 1-5 → 2-10
                
                # Save mood entry with reference to questionnaire
                cursor = conn.execute(
                    """
                    INSERT INTO mood_entries (user_id, mood_level_id, recorded_at, notes)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        st.session_state.user_id,
                        mood_score_10,
                        datetime.now().isoformat(),
                        f"Depression Assessment (Q{questionnaire_id}) - Structured: {structured_score}/5.0, LLM: {llm_score}/5.0, Final: {final_score}/5.0 | Text: {text_response[:150]}"
                    )
                )
                conn.commit()
                
                # Display results
                st.success("✅ Assessment submitted successfully!")
                
                st.markdown("---")
                st.subheader("📊 Your Assessment Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Structured Score", f"{structured_score}/5.0")
                    st.caption("Based on questionnaire responses")
                
                with col2:
                    st.metric("LLM Analysis", f"{llm_score}/5.0")
                    st.caption("AI text analysis")
                
                with col3:
                    st.metric("Final Score", f"{final_score}/5.0")
                    st.caption("Combined assessment")
                
                # Score interpretation
                st.markdown("---")
                if final_score <= 2.0:
                    st.error("⚠️ **Severe Concern**: Your responses indicate significant distress. Please consider reaching out to a mental health professional or crisis helpline immediately.")
                elif final_score <= 3.0:
                    st.warning("⚠️ **Moderate Concern**: Your responses show signs of depression. We recommend speaking with a healthcare provider.")
                elif final_score <= 4.0:
                    st.info("ℹ️ **Mild Concern**: Some negative indicators detected. Consider self-care activities and monitoring your mood.")
                else:
                    st.success("✅ **Positive Assessment**: Your responses indicate relatively good mental health. Keep up healthy habits!")
                
                # LLM explanation
                if llm_explanation:
                    with st.expander("🤖 AI Analysis Details"):
                        st.write(llm_explanation)
                
                # Mark form as submitted
                st.session_state.form_submitted = True
                        
        except Exception as e:
            st.error(f"Error saving assessment: {e}")
            st.exception(e)

# Show navigation buttons after successful submission (outside form context)
if st.session_state.form_submitted:
    st.markdown("---")
    st.subheader("What would you like to do next?")
    
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🏠 Return Home", key="nav_home", use_container_width=True):
            st.session_state.redirect_page = "pages/Home.py"
            st.rerun()
    with col_b:
        if st.button("📈 View Mood History", key="nav_mood", use_container_width=True):
            st.session_state.redirect_page = "pages/Mood Tracker.py"
            st.rerun()
