from turtledemo.nim import COLOR

import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode = False, model_complexity = 1, min_detection_confidence = 0.5)
mp_drawing = mp.solutions.drawing_utils

def load_video(video_path):
    video_path = "/videos/jav_1.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error, failed to open")

    return cap

def extract_frames(cap, interval = 10):
    frames = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not  ret:
            break
        if frame_count % interval == 0: #Extract every 10th frame
            frames.append(frame)
        frame_count += 1
    cap.release()

    return frames

def preprocess_frames(frames, target_size=(256, 256)):
    processed_frames = []
    for frame in frames:
        resized_frame = cv2.resize(frame, target_size)
        normalized_frame = resized_frame
        processed_frames.append(normalized_frame)

    return processed_frames

def extract_pose_keypoints(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #MediaPipe works with RGB not BGR
    #processing the frame
    results = pose.process(rgb_frame)
    if results.pose_landmarks:
        #Extract keypoints
        keypoints = [(lm.x, lm.y) for lm in results.pose_landmarks.landmark]

        return keypoints
    return None

def visualise_pose(frame, keypoints):
    annotated_frame = frame.copy()
    if keypoints:
        # Drawing of pose landmarks on the frame
        mp_drawing.draw_landmarks(
            annotated_frame,
            mp_pose.PoseLandmark(keypoints),
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color = (0, 255, 0), thickness=2, circle_radius= 2),
            mp_drawing.DrawingSpec(color = (255, 0, 0), thickness=2, circle_radius=2)
        )

        return annotated_frame

def compute_angle(a,b,c):
    #Angle between 3 points
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a- b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

def analyze_throw_technique(keypoint):
    #Compute elbow angle
    shoulder = keypoint[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    elbow = keypoint[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    wrist = keypoint[mp_pose.PoseLandmark.LEFT_WRIST.value]
    elbow_angle = compute_angle(shoulder, elbow, wrist)

    return {"elbow angle:" : elbow_angle}






