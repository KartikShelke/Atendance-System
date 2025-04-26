import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os
import mediapipe as mp
from datetime import datetime
from PIL import Image

# Load mediapipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Directory containing student images
STUDENTS_FOLDER = 'student'
ATTENDANCE_FILE = 'attendance.xlsx'

# Create Attendance File if not exist
if not os.path.exists(ATTENDANCE_FILE):
    df = pd.DataFrame(columns=['Name', 'Time'])
    df.to_excel(ATTENDANCE_FILE, index=False)

# Load student images
def load_student_images():
    images = []
    names = []
    for filename in os.listdir(STUDENTS_FOLDER):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(STUDENTS_FOLDER, filename)
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img_rgb)
            names.append(os.path.splitext(filename)[0])
    return images, names

# Match face with saved images (simple pixel difference check)
def match_face(face, student_images, student_names):
    face_resized = cv2.resize(face, (150,150))
    for idx, img in enumerate(student_images):
        img_resized = cv2.resize(img, (150,150))
        diff = np.mean(np.abs(face_resized - img_resized))
        if diff < 50:  # Threshold can be adjusted
            return student_names[idx]
    return None

# Mark attendance
def mark_attendance(name):
    df = pd.read_excel(ATTENDANCE_FILE)
    if name not in df['Name'].values:
        now = datetime.now()
        time_string = now.strftime('%H:%M:%S')
        df = pd.concat([df, pd.DataFrame([[name, time_string]], columns=['Name', 'Time'])])
        df.to_excel(ATTENDANCE_FILE, index=False)

# Streamlit App
st.title("Attendance Monitoring System 📸")

run = st.button('Start Camera')

if run:
    student_images, student_names = load_student_images()
    cap = cv2.VideoCapture(0)

    with mp_face_detection.FaceDetection(min_detection_confidence=0.7) as face_detection:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                    face_crop = frame_rgb[y:y+h, x:x+w]

                    name = match_face(face_crop, student_images, student_names)

                    if name:
                        mark_attendance(name)
                        cv2.putText(frame, f'{name}', (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    else:
                        cv2.putText(frame, 'Unknown', (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2)

            st.image(frame, channels="BGR")

    cap.release()
