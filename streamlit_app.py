import streamlit as st
from collections import defaultdict
import csv
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_option_menu import option_menu
import base64
from scipy.spatial.distance import euclidean

# Function to get image and encode it to base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()



# Function to set a background image
def set_bg_image():
    bin_str = get_base64_of_bin_file('background.png')  # replace with your filename
    page_bg_img = '''
    <style>
    .stApp {
      background-image: url("data:image/png;base64,%s");
      background-size: cover;
      background-position: center;
    }
    </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)

# Define the style for your custom box
custom_box_style = """
<style>
.custom-box {
    background-color: #27005d;  /* Custom background color */
    color: #ffffff;            /* Custom text color */
    padding: 10px;             /* Padding inside the box */
    border-style: solid;       /* Creates the solid border */
    border-radius: 10px;       /* Rounded corners */
    border-width: 4px;        /*  Creates border width */
    border-color: #aed2ff;     /* Colors the border */
    margin: 10px 0;            /* Margin around the box */
    box-shadow: 8px 8px 10px #27005d; /* Shadow effect: horizontal offset, vertical offset, blur radius, color */
}

.custom-box-2 {
    background-color: #9400ff;  /* Custom background color */
    color: #000000;            /* Custom text color */
    padding: 10px;             /* Padding inside the box */
    border-style: solid;       /* Creates the solid border */
    border-radius: 10px;       /* Rounded corners */
    border-width: 4px;        /*  Creates border width */
    border-color: #aed2ff;     /* Colors the border */
    margin: 10px 0;            /* Margin around the box */
    box-shadow: 8px 8px 10px #27005d; /* Shadow effect: horizontal offset, vertical offset, blur radius, color */
    text-align: center;
}
</style>


"""

# Insert the custom box style into the Streamlit app
st.markdown(custom_box_style, unsafe_allow_html=True)

set_bg_image()
st.markdown(
    """
    <style>
    .reportview-container .main .block-container{
        padding-top: 0rem;
        padding-bottom: 0rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    choose = option_menu("CrowdSense", ["Pause Camera", "Aktifkan Kamera", "Berikan Notif"],
                         icons=['pause', 'camera-video','bell'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
                             "container": {"padding": "5!important", "background-color": "#27005D", "border-style" : "solid", "border-width" : "4px", "border-color" : "#aed2ff"},
                             "icon": {"color": "orange", "font-size": "25px"},
                             "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px",
                                         "--hover-color": "#000000"},
                             "nav-link-selected": {"background-color": "#9400FF"},
                         }
                         )
    
title_text = f"<div class='custom-box-2'><h1>Crowdsense</h1></div>"



st.markdown(title_text, unsafe_allow_html=True)

# Create two columns for the placeholders
col1, col2 = st.columns(2)

# Constants for crowd behavior classification
FIGHTING_DISTANCE_THRESHOLD = 50
STAMPEDE_SPEED_THRESHOLD = 300
SOCIAL_DISTANCE = 50
ABNORMAL_ENERGY = 1866
ABNORMAL_THRESH = 0.66
ABNORMAL_MIN_PEOPLE = 5

# Function to get the center of the bounding box
def get_center(box):
    x1, y1, x2, y2 = box.xyxy[0]
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def format_movement_track(track):
    coordinates = [(int(x), int(y)) for (x, y) in track]
    flat_coordinates = [coord for pair in coordinates for coord in pair]
    return flat_coordinates

# Function to calculate trajectory length
def calculate_trajectory_length(movement_track):
    return sum(euclidean(movement_track[i], movement_track[i + 1]) for i in range(len(movement_track) - 1))

# Function to detect abnormal behavior based on speed
def detect_abnormal_behavior(movement_track, speed_threshold):
    for i in range(len(movement_track) - 1):
        speed = euclidean(movement_track[i], movement_track[i + 1])
        if speed > speed_threshold:
            return True
    return False

# Dropdown system for camera selection (no camera name fetching)
def get_available_cameras(max_checks=10):
    available_cameras = []
    for i in range(max_checks):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
        cap.release()
    return available_cameras

# Get the list of available camera indexes
available_cameras = get_available_cameras()

# Create a dropdown menu for the available cameras
camera_index = st.selectbox('Select Camera', available_cameras)

model = YOLO("best1.pt")
cap = cv2.VideoCapture(camera_index)

track_history = defaultdict(lambda: [])
track_meta = defaultdict(lambda: {"entry_time": None, "exit_time": None})
dot_image = np.zeros((720, 1280, 3), dtype=np.uint8)
restricted_entry = False  # Variable to store if restricted entry detected
abnormal_activity = False  # Variable to store if abnormal activity detected
violate_count = 0  # Variable to count social distance violations



frame_placeholder_annotated = col1.empty()  # Placeholder for displaying the annotated frame
frame_placeholder_dot = col2.empty()        # Placeholder for displaying the dot image

annotated_frame = None  # Initialize annotated_frame

while choose == "Aktifkan Kamera":
    success, frame = cap.read()
    if not success:
        break

    results = model.track(frame, persist=True)

    if results is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        annotated_frame = results[0].plot()  # Update annotated_frame

        current_detected_ids = set()
        for box, track_id in zip(boxes, track_ids):
            center_x, center_y = get_center(box)
            track_history[track_id].append((center_x, center_y))
            cv2.circle(dot_image, (center_x, center_y), 10, (0, 255, 0), -1)  # Draw centroid

            # Check for social distance violation
            for other_track_id, other_track in track_history.items():
                if track_id != other_track_id and len(other_track) > 1:
                    distance = euclidean(track_history[track_id][-1], other_track[-1])
                    if distance < SOCIAL_DISTANCE:
                        violate_count += 1

            # Check for abnormal behavior
            if detect_abnormal_behavior(track_history[track_id], STAMPEDE_SPEED_THRESHOLD):
                abnormal_activity = True

            current_detected_ids.add(track_id)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    center_x, center_y = get_center(box)
                    cv2.circle(dot_image, (center_x, center_y), 10, (0, 255, 0), -1)

    # Check if annotated_frame is not None before displaying
    if annotated_frame is not None:
        frame_placeholder_annotated.image(annotated_frame, channels="BGR", caption="Tracking", use_column_width=True)
    frame_placeholder_dot.image(dot_image, channels="BGR", caption="Dot Image", use_column_width=True)


cap.release()

# Display results
# Use the custom box style in the sidebar
violations_text = f"Social Distance Violations: {violate_count}"
abnormal_activity_text = "Abnormal Activity: DETECTED" if abnormal_activity else "Abnormal Activity: Not detected"

combined_text = f"<div class='custom-box'>{violations_text}<br>{abnormal_activity_text}</div>"


st.sidebar.markdown(combined_text, unsafe_allow_html=True)

# Export data as CSV
if st.button('Export data dalam CSV'):
    with open("track_data.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "entry_time", "exit_time", "Movement_tracks"])
        for track_id, track in track_history.items():
            flat_coordinates = format_movement_track(track)
            writer.writerow([track_id, track_meta[track_id]["entry_time"], track_meta[track_id]["exit_time"]] + flat_coordinates)

    st.success('CSV file generated!')

# Activate or deactivate camera based on user selection
if choose != "Aktifkan Kamera" and choose == "Pause Camera":
    st.session_state.camera_active = False
    
# Place this where you handle the button actions
if choose == "Berikan Notif":
    if abnormal_activity:
        st.error("ALERT: Abnormal Activity DETECTED!")  # Red alert box for detected abnormal activity
    else:
        st.info("No abnormal activity detected.")  # Blue info box for normal situations