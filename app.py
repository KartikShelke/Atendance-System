import cv2
import os
import pandas as pd
import streamlit as st
from datetime import datetime

# Path to the folder where student images are stored
STUDENTS_FOLDER = "Students"
ATTENDANCE_FILE = "attendance.csv"

# Load student images (using OpenCV)
def load_student_images():
    student_images = []
    student_names = []
    
    # Iterate through all images in the student folder
    for filename in os.listdir(STUDENTS_FOLDER):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(STUDENTS_FOLDER, filename)
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for recognition
            student_images.append(image_rgb)
            student_names.append(filename.split('.')[0])  # Assuming name is in filename
    return student_images, student_names

# Save attendance
def save_attendance(name):
    with open(ATTENDANCE_FILE, "a") as f:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{name},{current_time}\n")

# Face detection function using OpenCV
def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces

# Start video capture for live attendance
def mark_attendance():
    student_images, student_names = load_student_images()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Start capturing video from webcam (try different indices if 0 fails)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Camera not found or in use. Try restarting the app or check your device.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            # Crop the face from the frame
            face = frame[y:y+h, x:x+w]
            
            # Compare the detected face with student images (using a simple method for now)
            for student_image, student_name in zip(student_images, student_names):
                # Use template matching or other methods to compare faces (you can improve this part)
                # For now, assume it's a match if the face is detected
                save_attendance(student_name)
                cv2.putText(frame, f"Attendance marked for: {student_name}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Break after marking attendance once
                break
        
        # Display the video feed with face detection using Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB for Streamlit
        st.image(frame_rgb, channels="RGB", use_column_width=True)
        
        # Exit if the user presses the 'q' key (streamlit doesn't support key capture in the same way)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the webcam
    cap.release()

# Streamlit UI to mark attendance
st.title('Attendance System')
if st.button('Start Attendance'):
    mark_attendance()
