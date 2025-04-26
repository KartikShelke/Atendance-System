import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Initialize or load attendance
ATTENDANCE_FILE = "attendance.csv"

if not os.path.exists(ATTENDANCE_FILE):
    df = pd.DataFrame(columns=["Name", "Time"])
    df.to_csv(ATTENDANCE_FILE, index=False)

# Load known faces (fake for now, because we don't use face_recognition)
known_names = ["Person1", "Person2", "Person3"]  # Add your list of names

# Attendance marking function
def mark_attendance(name):
    df = pd.read_csv(ATTENDANCE_FILE)
    if name not in df['Name'].values:
        now = datetime.now()
        dt_string = now.strftime('%Y-%m-%d %H:%M:%S')
        new_entry = pd.DataFrame([[name, dt_string]], columns=["Name", "Time"])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)

# Dummy face detection function (because no face_recognition)
def detect_faces(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    return faces

# Main app
def main():
    st.title("Attendance System without face_recognition")

    run = st.checkbox('Run Camera')

    FRAME_WINDOW = st.image([])

    cap = None

    if run:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("❌ Camera not found or in use. Please check your device.")
            return

        st.success('✅ Camera started! Showing live feed...')
        
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame.")
                break

            faces = detect_faces(frame)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Dummy name (in real case, identify the person)
                name = known_names[0]  
                mark_attendance(name)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)

    else:
        st.warning('Camera is off.')

    if cap:
        cap.release()

    if st.button('Show Attendance'):
        df = pd.read_csv(ATTENDANCE_FILE)
        st.dataframe(df)

if __name__ == '__main__':
    main()
