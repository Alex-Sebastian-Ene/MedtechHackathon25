import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av 

class VideoProcessor:

    def recv(self, frame):
        frm = frame.to_ndarray(format= "bgr24")

        return av.VideoFrame.from_ndarray(frm, format='bgr24')
    
webrtc_streamer(
    key="exportArray",  # unique key for the streamer
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False}  # optional
)