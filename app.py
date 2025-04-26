import streamlit as st
import face_recognition
import numpy as np
import pandas as pd
import os
from datetime import datetime
from PIL import Image

# Load student images
path = 'Students'
images = []
studentNames = []
for filename in os.listdir(path):
    img = face_recognition.load_image_file(f'{path}/{filename}')
    images.append(img)
    studentNames.append(os.path.splitext(filename)[0])

# Encode the student images
def findEncodings(images):
    encodeList = []
    for img in images:
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)

# Function to mark attendance
def markAttendance(name):
    now = datetime.now()
    dtString = now.strftime('%Y-%m-%d %H:%M:%S')
    new_entry = pd.DataFrame([[name, dtString]], columns=["Name", "Timestamp"])
    
    if os.path.exists('Attendance.xlsx'):
        existing = pd.read_excel('Attendance.xlsx')
        updated = pd.concat([existing, new_entry], ignore_index=True)
        updated.to_excel('Attendance.xlsx', index=False)
    else:
        new_entry.to_excel('Attendance.xlsx', index=False)

# Streamlit UI
st.title("🎯 Face Recognition Attendance System")

uploaded_image = st.camera_input("Capture Your Image")

if uploaded_image:
    img = Image.open(uploaded_image)
    img_np = np.array(img)

    facesCurFrame = face_recognition.face_locations(img_np)
    encodesCurFrame = face_recognition.face_encodings(img_np, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = studentNames[matchIndex]
            st.success(f"✅ Attendance marked for {name}")
            markAttendance(name)
        else:
            st.error("❌ No match found. Please try again.")
