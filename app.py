import os
import subprocess
import sys

# Ensure face_recognition is installed
subprocess.check_call([sys.executable, "-m", "pip", "install", "face_recognition"])

# Then proceed with the imports
import cv2
import face_recognition
import pandas as pd
import streamlit as st
from datetime import datetime


# Define the folder to store student images
STUDENTS_FOLDER = 'Students'

# Check if the folder exists, if not create it
if not os.path.exists(STUDENTS_FOLDER):
    os.makedirs(STUDENTS_FOLDER)

# Function to load student images and names
def load_student_images():
    student_images = []
    student_names = []
    
    for filename in os.listdir(STUDENTS_FOLDER):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # Load image
            image_path = os.path.join(STUDENTS_FOLDER, filename)
            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Extract face encoding
            face_encoding = face_recognition.face_encodings(img_rgb)[0]
            
            # Append to lists
            student_images.append(face_encoding)
            student_names.append(filename.split('.')[0])  # Assuming filename is the name of the student

    return student_images, student_names

# Function to mark attendance
def mark_attendance(student_name):
    # Get the current time and date
    now = datetime.now()
    dt_string = now.strftime('%Y-%m-%d %H:%M:%S')

    # Load existing attendance file if it exists
    if os.path.exists('Attendance.xlsx'):
        df = pd.read_excel('Attendance.xlsx')
    else:
        df = pd.DataFrame(columns=["Name", "Time"])

    # Check if the student has already marked attendance
    if student_name not in df['Name'].values:
        # Add new record to the DataFrame
        df = df.append({"Name": student_name, "Time": dt_string}, ignore_index=True)
        # Save to Excel file
        df.to_excel('Attendance.xlsx', index=False)

# Streamlit UI components
st.title("Attendance Monitoring System")
st.write("This app uses face recognition to mark student attendance.")

# Load student images
student_images, student_names = load_student_images()

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture image from webcam")
        break

    # Convert the image to RGB (OpenCV uses BGR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all face locations and face encodings in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Check if the detected face matches any student
        matches = face_recognition.compare_faces(student_images, face_encoding)
        
        if True in matches:
            first_match_index = matches.index(True)
            student_name = student_names[first_match_index]
            mark_attendance(student_name)
            st.success(f"Attendance marked for {student_name}")
            break

    # Show the webcam feed
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame_rgb, channels="RGB", use_column_width=True)

# Release the webcam
cap.release()
cv2.destroyAllWindows()
