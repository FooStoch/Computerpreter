import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import pickle
import numpy as np
import mediapipe as mp
import torch
import os
import whisper
import streamlit.components.v1 as components
import tempfile
from io import BytesIO

# Page config
st.set_page_config(layout="wide")
st.title("Computerpreter")


left_col, right_col = st.columns([1, 1])

# if "history_finger" not in st.session_state:
#     st.session_state.history_finger = []
# if "result_words" not in st.session_state:
#     st.session_state.result_words = []

# Load static fingerspelling model
@st.cache_resource
def load_finger_model():
    model = pickle.load(open("model.p", "rb"))["model"]
    return model
finger_model = load_finger_model()

# Load dynamic ASL model module
@st.cache_resource
def load_dynamic_module():
    import asl_inference
    return asl_inference
asl = load_dynamic_module()

# Setup Mediapipe for static
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.3,
                       min_tracking_confidence=0.3)
labels_dict = {i: chr(ord('A') + i) for i in range(26)}

# Processor for static fingerspelling
#hf = []
def create_finger_processor():
    class FingerProcessor(VideoProcessorBase):
        def __init__(self):
            self.processor = hands
            #self.raw_history = []
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            try:
                img = frame.to_ndarray(format="bgr24")
                H, W, _ = img.shape
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                res = self.processor.process(rgb)
                if res.multi_hand_landmarks:
                    for hl in res.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            img, hl, mp_hands.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connections_style()
                        )
                        xs = [lm.x for lm in hl.landmark]
                        ys = [lm.y for lm in hl.landmark]
                        data = []
                        for x, y in zip(xs, ys):
                            data += [x - min(xs), y - min(ys)]
                        char = ''
                        if len(data) == 42:
                            p = finger_model.predict([np.array(data)])[0]
                            char = labels_dict[int(p)]
                            #st.session_state.history_finger.append(char)
                            #self.raw_history.append(char)
                            #hf.append(char)
                        x1, y1 = int(min(xs)*W)-10, int(min(ys)*H)-10
                        x2, y2 = int(max(xs)*W)+10, int(max(ys)*H)+10
                        cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,0), 4)
                        cv2.putText(
                            img, char, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                            1.3, (0,0,0), 3, cv2.LINE_AA
                        )
                

                return av.VideoFrame.from_ndarray(img, format="bgr24")
            except Exception as e:
                return frame   
    return FingerProcessor


# Processor for dynamic ASL sequence detection
def create_dynamic_processor():
    class DynamicProcessor(VideoProcessorBase):
        def __init__(self):
            self.holistic = asl.mp_holistic.Holistic(
                static_image_mode=False,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
            self.buffer = []
            self.max_frames = 30
            self.last_text = ""
            self.display_count = 0
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            try:
                img = frame.to_ndarray(format="bgr24")
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.holistic.process(rgb)
                landmarks = asl.extract_landmarks(img)
                if landmarks is not None:
                    self.buffer.append(landmarks)
                    img = asl.draw_landmarks(img, landmarks)
                # When enough frames collected, predict and set text
                if len(self.buffer) >= self.max_frames:
                    sign, conf = asl.predict_sign(self.buffer, asl.model, asl.device)
                    self.last_text = f"{sign} ({conf*100:.1f}%)"
                    self.display_count = self.max_frames  # show text for next max_frames frames
                    self.buffer.clear()
                # Draw last_text if within display window
                if self.display_count > 0:
                    cv2.putText(
                        img, self.last_text,
                        (10, img.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 0), 2, cv2.LINE_AA
                    )
                    self.display_count -= 1
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            except Exception as e:
                return frame 
    return DynamicProcessor

# UI select mode
with left_col: 
    mode = st.selectbox("Select mode:", ["Fingerspelling", "Dynamic Sign"])

    # STUN server configuration
    rtc_conf = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

    # def process_finger_history():
    #     if not st.session_state.history_finger:
    #         return
        
    #     window_size = 10
    #     threshold = 6
    #     result_finger = []
    #     prev_main = None

    #     for i in range(len(st.session_state.history_finger) - window_size + 1):
    #         window = st.session_state.history_finger[i:i+window_size]
    #         counts = {}
    #         for letter in window:
    #             counts[letter] = counts.get(letter, 0) + 1
    #         main_letter = max(counts, key=counts.get)
    #         if counts[main_letter] >= threshold:
    #             if main_letter != prev_main:
    #                 result_finger.append(main_letter)
    #                 prev_main = main_letter

    #     if result_finger:
    #         result_finger_string = ''.join(result_finger)
    #         st.session_state.result_words.append(result_finger_string)
    #         st.session_state.history_finger = []  # Clear the history after processing

    if mode == "Fingerspelling":
        webrtc_ctx_finger = webrtc_streamer(
            key="finger",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=create_finger_processor(),
            media_stream_constraints={
                "video": {"frameRate": {"ideal": 10, "max": 15}},
                "audio": False
            },
            async_processing=True,
            rtc_configuration=rtc_conf, 
        )

        # if not webrtc_ctx_finger.video_processor and st.session_state.history_finger is not None:
        #     window_size = 10
        #     threshold = 6
        #     result_finger = []
        #     prev_main = None

        #     for i in range(len(st.session_state.history_finger) - window_size + 1):
        #         window = st.session_state.history_finger[i:i+window_size]
        #         counts = {}
        #         for letter in window:
        #             counts[letter] = counts.get(letter, 0) + 1
        #         main_letter = max(counts, key=counts.get)
        #         if counts[main_letter] >= threshold:
        #             if main_letter != prev_main:
        #                 result_finger.append(main_letter)
        #                 prev_main = main_letter


        #     result_finger_string = ''.join(result_finger)
        #     # st.session_state.history.append(result_finger_string)
        #     st.text(result_finger_string)
            #hf = []
            # st.session_state.history_finger.clear()
        #     proc = webrtc_ctx_finger.video_processor
        #     if proc.raw_history:
        #         st.session_state.history_finger.append(proc.raw_history)
        #         proc.raw_history.clear()
                
        #         # Process history if we have enough frames
        #         if len(st.session_state.history_finger) >= 3:  # Adjust this threshold as needed
        #             process_finger_history()
        
        # if st.session_state.result_words is not None:
        #     for word in st.session_state.result_words:
        #         st.text(word)
        # if st.button("Clear Words"):
        #     st.session_state.result_words = []
        #     st.session_state.history_finger = []
            # raw_history is your per-frame list of detected chars
        #st.write("Detected so far:", st.session_state.history_finger)
        #proc = webrtc_ctx_finger.video_processor
        #if proc is not None:
            # move everything from proc.raw_history into session_state
            #st.session_state.history_finger += proc.raw_history
            #proc.raw_history.clear()
        #hf = st.session_state.history_finger
        #if hf:
            #window_size = 10
            #threshold = 6
            #result_finger = []
            #prev_main = None

            #for i in range(len(hf) - window_size + 1):
                #window = hf[i:i+window_size]
                #counts = {}
                #for letter in window:
                    #counts[letter] = counts.get(letter, 0) + 1
                #main_letter = max(counts, key=counts.get)
                #if counts[main_letter] >= threshold:
                    #if main_letter != prev_main:
                        #result_finger.append(main_letter)
                        #prev_main = main_letter


            #result_finger_string = ''.join(result_finger)
            #st.session_state.history.append(result_finger_string)
            #st.text(result_finger_string)
            #st.session_state.history_finger.clear()

    else:
        webrtc_streamer(
            key="dynamic",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=create_dynamic_processor(),
            media_stream_constraints={
                "video": {"frameRate": {"ideal": 10, "max": 15}},
                "audio": False
            },
            async_processing=True,
            rtc_configuration=rtc_conf
        )
    


with right_col: 

    st_audiorec = components.declare_component(
        "st_audiorec", path="st_audiorec/frontend/build"
    )


    record_result = st_audiorec()

    wav_bytes = None

    if isinstance(record_result, dict) and "arr" in record_result:
        # Stefan's unpacking: record_result['arr'] is a map of {index: byte_value}
        with st.spinner("processing audio…"):
            ind, raw = zip(*record_result["arr"].items())
            ind = np.array(ind, dtype=int)
            raw = np.array(raw, dtype=int)
            sorted_bytes = raw[ind]                      # reorder by index
            # build a bytestream
            stream = BytesIO(bytearray(int(v) & 0xFF for v in sorted_bytes))
            wav_bytes = stream.read()

    elif isinstance(record_result, (bytes, bytearray)):
        # in case the component ever returns raw bytes directly
        wav_bytes = bytes(record_result)

    # save into session_state
    st.session_state.audio_data = wav_bytes

    model = whisper.load_model("base")

    if "history" not in st.session_state:
        st.session_state.history = []


    if st.button("Transcribe Audio"):
        if st.session_state.audio_data is None:
            st.error("No recording found!")
        else:
            # write to a temp file and pass that path to Whisper
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(st.session_state.audio_data)
            tmp.flush()
            tmp_path = tmp.name
            tmp.close()

            model = whisper.load_model("base")
            #st.success("Transcribing Audio…")
            transcription = model.transcribe(tmp_path)
            #st.success("Done!")
            st.session_state.history.append(transcription["text"])
            #st.experimental_rerun()

if st.button("Clear History"):
    st.session_state.history.clear()

for msg in st.session_state.history:
    st.chat_message("user").write(msg)

