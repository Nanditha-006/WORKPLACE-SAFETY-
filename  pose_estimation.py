import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Distance Calculation
def calculate_distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

# Angle Calculation
def calculate_angle(a, b, c):
    angle = math.degrees(
        math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
    )
    angle = abs(angle)
    return angle if angle <= 180 else 360 - angle

# Pose Classification Logic
def classify_pose(landmarks):
    # Key Landmarks
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

    # Angles & Distances
    left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
    torso_tilt = abs(left_shoulder.y - left_hip.y) + abs(right_shoulder.y - right_hip.y)
    wrist_above_head = left_wrist.y < nose.y and right_wrist.y < nose.y
    hand_near_face = min(calculate_distance(left_wrist, nose), calculate_distance(right_wrist, nose)) < 0.1
    hip_height_difference = abs(left_hip.y - right_hip.y)

    # Climbing Detection Criteria
    legs_asymmetry = abs(left_knee.y - right_knee.y) > 0.15
    arms_asymmetry = abs(left_wrist.y - right_wrist.y) > 0.15
    climbing_posture = (legs_asymmetry and arms_asymmetry and (wrist_above_head or torso_tilt > 0.1))

    # Pose Classification
    if climbing_posture:
        return "Climbing"
    elif wrist_above_head:
        return "Raising Hands"
    elif hand_near_face:
        return "Touching Face"
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

# Video Capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        pose_class = classify_pose(landmarks)

        # Draw Landmarks
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )

        # Display Pose
        cv2.putText(frame, pose_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('Pose Estimation', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
 