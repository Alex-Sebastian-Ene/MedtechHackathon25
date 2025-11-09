import time
from typing import Any, Optional
from datetime import datetime

import av
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_webrtc import webrtc_streamer

from medtech_hackathon25.ml.vision.emotion_recognition import (
    FaceEmotionRecognizer,
    score_results,
)
from databases.database import get_connection, save_video_recording


MAX_RECORD_SECONDS = 10
DEFAULT_STATUS_MESSAGE = "Recording… auto stop at 10 seconds."


class TimedVideoProcessor:
    """Capture frames until the 10-second limit is reached."""

    def __init__(self) -> None:
        self.start_time: float | None = None
        self.finished = False
        self.last_frame: Optional[np.ndarray] = None

    def recv(self, frame):
        if self.start_time is None:
            self.start_time = time.time()

        img = frame.to_ndarray(format="bgr24")
        self.last_frame = img.copy()
        elapsed = time.time() - self.start_time

        if elapsed >= MAX_RECORD_SECONDS:
            self.finished = True

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def _init_session_state() -> None:
    defaults = {
        "desired_playing_state": None,
        "stop_message": "",
        "analysis_result": None,
        "analysis_error": None,
        "analysis_pending": False,
        "recorded_frame": None,
        "record_start_time": None,
        "emotion_history": [],
        "last_analysis_time": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


@st.cache_resource(show_spinner="Loading emotion recognition models...")
def _get_emotion_recognizer() -> FaceEmotionRecognizer:
    """Cache the emotion recognizer to avoid reloading models on every page refresh."""
    return FaceEmotionRecognizer()


def _analyze_emotions_locally(frame: np.ndarray) -> Any:
    recognizer = _get_emotion_recognizer()
    detections = recognizer.analyze_frame(frame)
    mood = score_results(detections)
    return {"mood": mood, "detections": detections}


def _save_mood_to_database(user_id: int, mood_score: float, primary_emotion: str, notes: str = "") -> bool:
    """Save the mood entry to the database."""
    try:
        # Round mood score to nearest integer for mood_level_id (1-10)
        mood_level_id = max(1, min(10, round(mood_score)))
        
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO mood_entries (user_id, mood_level_id, recorded_at, notes)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, mood_level_id, datetime.now().isoformat(), notes)
            )
            conn.commit()
        return True
    except Exception as e:
        st.error(f"Failed to save mood data: {str(e)}")
        return False


def _reset_recording_state(auto_start: bool = False) -> None:
    st.session_state.analysis_result = None
    st.session_state.analysis_error = None
    st.session_state.analysis_pending = False
    st.session_state.recorded_frame = None
    st.session_state.stop_message = ""
    st.session_state.desired_playing_state = True if auto_start else None
    st.session_state.emotion_history = []
    st.session_state.last_analysis_time = None
    st.session_state.analysis_errors = []
    st.session_state.mood_saved = False


_init_session_state()

# Check if user is logged in
if "user_id" not in st.session_state or st.session_state.user_id is None:
    st.warning("Please log in to use the emotion scanner.")
    if st.button("Go to Login"):
        st.switch_page("login.py")
    st.stop()

# Display page title and user info
st.title("🎥 Emotion Video Check-Up")
st.caption(f"👤 Logged in as: {st.session_state.get('username', 'Unknown')}")

# Add info about the scan
with st.expander("ℹ️ How it works", expanded=False):
    st.markdown("""
    - **10-second scan**: The camera will record for 10 seconds
    - **Real-time analysis**: Your emotions are analyzed multiple times per second
    - **Mood score**: Get an overall mood rating from 1-10
    - **Automatic save**: Results are saved to your mood history
    - **Privacy**: All processing happens on your device
    """)

if st.session_state.analysis_result or st.session_state.analysis_error:
    if st.button("Start New Scan", type="primary"):
        _reset_recording_state(auto_start=True)
        st.rerun()

st.markdown("""
    <style>
    .status-text {
        padding: 12px;
        border-radius: 8px;
        font-size: 16px;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# Create a container with fixed positioning
recording_container = st.container()

with recording_container:
    # Video component centered
    ctx = webrtc_streamer(
        key="exportArray",
        video_processor_factory=TimedVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        desired_playing_state=st.session_state.desired_playing_state,
    )
    
    # Status placeholder below video - always visible with fixed size
    status_container = st.container()
    with status_container:
        status_placeholder = st.empty()


if ctx:
    if ctx.state.playing and st.session_state.desired_playing_state is True:
        # Reset desired state once playback has started to avoid repeated triggers.
        st.session_state.desired_playing_state = None
    if not ctx.state.playing and st.session_state.desired_playing_state is False:
        # Clear the stop signal after the stream has halted.
        st.session_state.desired_playing_state = None


processor = ctx.video_processor if ctx else None

if ctx and ctx.state.playing:
    now = time.time()
    if st.session_state.record_start_time is None:
        st.session_state.record_start_time = now
        st.session_state.last_analysis_time = now

    elapsed = now - st.session_state.record_start_time

    # Run emotion analysis every 0.2 seconds (reduced frequency for better performance)
    if processor and processor.last_frame is not None:
        last_analysis = st.session_state.last_analysis_time if st.session_state.last_analysis_time is not None else now - 0.3
        if now - last_analysis >= 0.2:  # Reduced from 0.1 to 0.2 for better performance
            try:
                # Lazy load the recognizer only when needed
                recognizer = _get_emotion_recognizer()
                from PIL import Image
                rgb = cv2.cvtColor(processor.last_frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
                detections = recognizer._detect_faces(image)
                
                detailed_results = []
                if detections:
                    for detected in detections:
                        face_tensor = detected.tensor
                        # Get emotion classification with all probabilities
                        pil_face = Image.fromarray((face_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype('uint8'))
                        inputs = recognizer.emotion_model.processor(images=pil_face, return_tensors="pt").to(recognizer.device)
                        outputs = recognizer.emotion_model.model(**inputs)
                        probs = outputs.logits.softmax(dim=-1).squeeze(0)
                        
                        # Create dict with all emotion scores
                        emotion_scores = {}
                        for idx, label in enumerate(recognizer.emotion_model.labels):
                            emotion_scores[label] = float(probs[idx])
                        
                        top_emotion = max(emotion_scores.items(), key=lambda x: x[1])
                        
                        detailed_results.append({
                            "box": detected.box.tolist(),
                            "detection_confidence": detected.probability,
                            "emotion": top_emotion[0],
                            "emotion_confidence": top_emotion[1],
                            "all_emotion_scores": emotion_scores
                        })
                    
                    mood = score_results(detailed_results)
                    
                    st.session_state.emotion_history.append({
                        "timestamp": elapsed,
                        "mood": mood,
                        "detections": detailed_results
                    })
                st.session_state.last_analysis_time = now
            except Exception as e:
                # Log the error but continue recording
                if 'analysis_errors' not in st.session_state:
                    st.session_state.analysis_errors = []
                st.session_state.analysis_errors.append(f"Analysis error at {elapsed:.1f}s: {str(e)}")
                import traceback
                st.session_state.analysis_errors.append(traceback.format_exc())

    if elapsed >= MAX_RECORD_SECONDS and not st.session_state.analysis_pending:
        frame_copy = processor.last_frame.copy() if processor and processor.last_frame is not None else None
        st.session_state.desired_playing_state = False
        st.session_state.stop_message = "Stop"
        st.session_state.recorded_frame = frame_copy
        st.session_state.analysis_pending = frame_copy is not None
        status_placeholder.success(st.session_state.stop_message)
        st.rerun()
    elif st.session_state.analysis_pending:
        status_placeholder.markdown(f'<div class="status-text" style="background-color: #e3f2fd; color: #1565c0; min-height: 50px; width: 100%; display: flex; align-items: center; justify-content: center;">Processing...</div>', unsafe_allow_html=True)
    else:
        analysis_count = len(st.session_state.emotion_history)
        status_placeholder.markdown(f'<div class="status-text" style="background-color: #e8f5e9; color: #2e7d32; min-height: 50px; width: 100%; display: flex; align-items: center; justify-content: center;">{elapsed:.1f}s | {analysis_count} frames</div>', unsafe_allow_html=True)
        time.sleep(0.1)  # Reduced rerun frequency from 144fps to 10fps for better performance
        st.rerun()
else:
    st.session_state.record_start_time = None
    if st.session_state.stop_message:
        status_placeholder.markdown(f'<div class="status-text" style="background-color: #e8f5e9; color: #2e7d32; min-height: 50px; width: 100%; display: flex; align-items: center; justify-content: center;">Complete</div>', unsafe_allow_html=True)
    else:
        status_placeholder.markdown(f'<div class="status-text" style="background-color: #f5f5f5; color: #616161; min-height: 50px; width: 100%; display: flex; align-items: center; justify-content: center;">Ready</div>', unsafe_allow_html=True)

    if st.session_state.analysis_pending:
        if st.session_state.emotion_history:
            # Use the emotion history data we already collected
            avg_mood = sum(entry["mood"] for entry in st.session_state.emotion_history) / len(st.session_state.emotion_history)
            
            # Aggregate all emotions from the recording
            emotion_counts = {}
            total_confidence = {}
            for entry in st.session_state.emotion_history:
                for detection in entry["detections"]:
                    if 'all_emotion_scores' in detection:
                        for emotion, score in detection['all_emotion_scores'].items():
                            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + score
                            total_confidence[emotion] = total_confidence.get(emotion, 0) + 1
            
            # Get the most frequent emotion
            if emotion_counts:
                avg_emotions = {em: emotion_counts[em] / total_confidence[em] for em in emotion_counts}
                top_emotion = max(avg_emotions.items(), key=lambda x: x[1])
                
                st.session_state.analysis_result = {
                    "mood": round(avg_mood, 1),
                    "detections": [{
                        "emotion": top_emotion[0],
                        "emotion_confidence": top_emotion[1],
                        "all_emotion_scores": avg_emotions
                    }]
                }
            else:
                st.session_state.analysis_result = {"mood": round(avg_mood, 1), "detections": []}
        else:
            st.session_state.analysis_error = "No analysis data collected."
            st.session_state.analysis_result = None
        
        st.session_state.analysis_pending = False


if st.session_state.analysis_result:
    result = st.session_state.analysis_result
    
    # Auto-save to database on first display
    if "mood_saved" not in st.session_state:
        st.session_state.mood_saved = False
    
    if not st.session_state.mood_saved and st.session_state.user_id:
        mood_score = result.get('mood', 5.0)
        detections = result.get("detections", [])
        primary_emotion = detections[0].get('emotion', 'unknown') if detections and 'emotion' in detections[0] else 'unknown'
        
        # Create a summary note with emotion details
        frames_analyzed = len(st.session_state.emotion_history) if st.session_state.emotion_history else 0
        notes = f"Emotion scan: {primary_emotion} ({frames_analyzed} frames analyzed)"
        
        # Save mood entry
        mood_saved = _save_mood_to_database(st.session_state.user_id, mood_score, primary_emotion, notes)
        
        # Save full video recording with all emotion frames
        if mood_saved and st.session_state.emotion_history:
            try:
                recording_id = save_video_recording(
                    user_id=st.session_state.user_id,
                    duration_seconds=MAX_RECORD_SECONDS,
                    mood_score=mood_score,
                    primary_emotion=primary_emotion,
                    frame_count=frames_analyzed,
                    emotion_history=st.session_state.emotion_history,
                    metadata=f"Analysis interval: 0.1s, Total detections: {frames_analyzed}"
                )
                st.session_state.mood_saved = True
                st.session_state.last_recording_id = recording_id
                st.success(f"✓ Mood data and video recording saved (ID: {recording_id})")
            except Exception as e:
                st.session_state.mood_saved = True
                st.warning(f"Mood saved, but video recording failed: {str(e)}")
        elif mood_saved:
            st.session_state.mood_saved = True
            st.success("✓ Mood data saved to your profile")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Mood Score", f"{result.get('mood')}/10")
    with col2:
        detections = result.get("detections", [])
        if detections and 'emotion' in detections[0]:
            st.metric("Primary Emotion", 
                     detections[0].get('emotion', 'unknown').title(), 
                     f"{detections[0].get('emotion_confidence', 0):.0%} confidence")
            
elif st.session_state.analysis_error:
    st.error(st.session_state.analysis_error)

if st.session_state.emotion_history:
    st.divider()
    st.subheader("Analysis Summary")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Frames Analyzed", len(st.session_state.emotion_history))
    with col2:
        avg_mood = sum(entry["mood"] for entry in st.session_state.emotion_history) / len(st.session_state.emotion_history)
        st.metric("Average Mood", f"{avg_mood:.1f}/10")
    
    st.divider()
    
    timeline_data = []
    for entry in st.session_state.emotion_history:
        timeline_data.append({
            "Time": entry["timestamp"],
            "Mood": entry["mood"]
        })
    
    df_timeline = pd.DataFrame(timeline_data)
    
    fig_timeline = px.line(
        df_timeline, 
        x="Time", 
        y="Mood",
        title="Mood Over Time",
        markers=True
    )
    
    fig_timeline.update_traces(fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.1)', line_color='rgb(59, 130, 246)')
    fig_timeline.update_layout(
        yaxis_range=[0, 10],
        showlegend=False,
        height=350
    )
    
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    emotion_timeline = []
    for entry in st.session_state.emotion_history:
        for detection in entry["detections"]:
            if 'all_emotion_scores' in detection:
                for emotion, score in detection['all_emotion_scores'].items():
                    emotion_timeline.append({
                        "Time": entry["timestamp"],
                        "Emotion": emotion.title(),
                        "Probability": score * 100
                    })
    
    if emotion_timeline:
        df_emotion = pd.DataFrame(emotion_timeline)
        
        fig_emotion = px.area(
            df_emotion,
            x="Time",
            y="Probability",
            color="Emotion",
            title="Emotion Distribution",
        )
        
        fig_emotion.update_layout(
            yaxis_range=[0, 100],
            height=350
        )
        
        st.plotly_chart(fig_emotion, use_container_width=True)
    
    st.divider()
    
    with st.expander("View Detailed Breakdown", expanded=False):
        for entry in st.session_state.emotion_history[-5:]:
            st.write(f"**{entry['timestamp']:.1f}s** — Mood: {entry['mood']}/10")
            for detection in entry["detections"]:
                if 'all_emotion_scores' in detection:
                    scores = detection['all_emotion_scores']
                    sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    emotion_text = " • ".join([f"{em.title()}: {sc:.0%}" for em, sc in sorted_emotions[:3]])
                    st.caption(emotion_text)
            st.write("")