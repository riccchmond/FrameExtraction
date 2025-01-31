from turtledemo.nim import COLOR

import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode = False, model_complexity = 1, min_detection_confidence = 0.5)
mp_drawing = mp.solutions.drawing_utils

def main():
    video_path = "/videos/jav_1.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error, failed to open")

    return cap