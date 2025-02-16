import streamlit as st
from ultralytics import YOLO
import mediapipe as mp
import cv2
import math

# Load YOLOv8 model
model = YOLO('/Users/nandhu/Desktop/final1/best.pt')  # Replace with your model path

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define PPE categories
PPE_CATEGORIES = ['Gloves', 'Mask', 'Goggles', 'lab coat', 'coverall','no mask','no labcoat','no goggles']

# Function to calculate distance between two points
def calculate_distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    angle = math.degrees(
        math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
    )
    angle = abs(angle)
    return angle if angle <= 180 else 360 - angle

# Function to classify posture based on landmarks
def classify_posture(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

    left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
    torso_tilt = abs(left_shoulder.y - left_hip.y) + abs(right_shoulder.y - right_hip.y)
    wrist_above_head = left_wrist.y < nose.y or right_wrist.y < nose.y
    hand_near_face = calculate_distance(left_wrist, nose) < 0.1 or calculate_distance(right_wrist, nose) < 0.1
    hip_height_difference = abs(left_hip.y - right_hip.y)
    touching_others = calculate_distance(left_wrist, right_wrist) < 0.1

    if wrist_above_head:
        return "Raising Hands"
    elif hand_near_face:
        return "Touching Face"
    elif touching_others:
        return "Touching Others"
    elif (left_knee_angle < 90 or right_knee_angle < 90) and torso_tilt > 0.1:
        return "Bending"
    elif left_hip.y > left_knee.y and right_hip.y > right_knee.y:
        return "Sitting"
    elif hip_height_difference > 0.1:
        return "Leaning"
    elif (left_ankle.y < left_knee.y and right_ankle.y < right_knee.y) and torso_tilt < 0.1:
        return "Running"
    else:
        return "Standing"

# Function to detect PPE
def detect_ppe(results):
    detected_ppe = []
    for result in results:
        for box in result.boxes:
            cls = model.names[int(box.cls[0])]
            if cls in PPE_CATEGORIES:
                detected_ppe.append(cls)
    return detected_ppe

# Streamlit app
st.title("PPE Detection and Pose Estimation")
st.write("Upload a video to detect PPE and estimate poses.")

# Upload video file
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save the uploaded video to a temporary file
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Open the video file
    cap = cv2.VideoCapture("temp_video.mp4")

    # Display video and process frames
    stframe = st.empty()

    # Frame skipping and downscaling
    frame_skip = 2  # Process every 2nd frame
    scale_percent = 50  # Reduce frame size to 50%
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # Skip this frame

        # Downscale the frame
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        frame = cv2.resize(frame, (width, height))

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run PPE detection
        results = model.predict(source=frame, conf=0.5)
        detected_ppe = detect_ppe(results)

        # Run pose estimation
        mp_results = pose.process(frame_rgb)

        # Draw PPE detection results
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = model.names[int(box.cls[0])]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{cls} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Draw pose estimation results
        if mp_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                mp_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

            # Classify posture
            posture = classify_posture(mp_results.pose_landmarks.landmark)
            cv2.putText(frame, f"Posture: {posture}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display detected PPE
        if detected_ppe:
            cv2.putText(frame, f"PPE Detected: {', '.join(detected_ppe)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the frame in Streamlit
        stframe.image(frame, channels="BGR", use_column_width=True)

    cap.release()