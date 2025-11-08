import time

import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av 


MAX_RECORD_SECONDS = 10
TIMER_TEXT_POSITION = (16, 36)
TIMER_FONT = cv2.FONT_HERSHEY_SIMPLEX
TIMER_FONT_SCALE = 1.0
TIMER_COLOR = (0, 255, 0)
TIMER_THICKNESS = 2


class TimedVideoProcessor:
    """Overlay elapsed time on frames and flag when the 10s limit is reached."""

    def __init__(self) -> None:
        self.start_time: float | None = None
        self.finished = False

    def recv(self, frame):
        if self.start_time is None:
            self.start_time = time.time()

        img = frame.to_ndarray(format="bgr24")
        elapsed = time.time() - self.start_time

        cv2.putText(
            img,
            f"{elapsed:.1f}s / {MAX_RECORD_SECONDS}s",
            TIMER_TEXT_POSITION,
            TIMER_FONT,
            TIMER_FONT_SCALE,
            TIMER_COLOR,
            TIMER_THICKNESS,
        )

        if elapsed >= MAX_RECORD_SECONDS:
            self.finished = True

        return av.VideoFrame.from_ndarray(img, format="bgr24")


if "record_start_time" not in st.session_state:
    st.session_state.record_start_time = None
if "force_stop" not in st.session_state:
    st.session_state.force_stop = False
if "stop_message" not in st.session_state:
    st.session_state.stop_message = ""

status_placeholder = st.empty()

desired_playing_state = False if st.session_state.force_stop else None

ctx = webrtc_streamer(
    key="exportArray",  # unique key for the streamer
    video_processor_factory=TimedVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},  # optional
    desired_playing_state=desired_playing_state,
)


def request_stop(message: str) -> None:
    st.session_state.force_stop = True
    st.session_state.stop_message = message
    st.rerun()


if ctx and ctx.state.playing:
    if st.session_state.record_start_time is None:
        st.session_state.record_start_time = time.time()

    processor = ctx.video_processor
    if processor and processor.finished and not st.session_state.force_stop:
        request_stop("Recording stopped automatically after 10 seconds.")
    else:
        status_placeholder.info("Recording… will stop automatically at 10 seconds.")
else:
    if st.session_state.record_start_time is not None:
        message = st.session_state.stop_message or "Recording stopped."
        status_placeholder.success(message)
    st.session_state.record_start_time = None
    st.session_state.force_stop = False
    st.session_state.stop_message = ""