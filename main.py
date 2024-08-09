import cv2
import mediapipe as mp
import streamlit as st
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh( min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_faces=2, static_image_mode=False)

st.title("SNAP")
st.sidebar.title("Choose Your Filter")
filter_option = st.sidebar.selectbox("Select a filter", ["None", "Sunglasses"])

video_capture = cv2.VideoCapture(0)

img = st.empty()

sunglasses = cv2.imread('sunglasses.png', -1)

def overlay_filter(frame, filter_img, landmarks, points_idx, scale):
    h, w, _ = frame.shape
    
    x1, y1 = int(landmarks[points_idx[0]].x * w), int(landmarks[points_idx[0]].y * h)
    x2, y2 = int(landmarks[points_idx[1]].x * w), int(landmarks[points_idx[1]].y * h)
    
    filter_width = int(np.linalg.norm([x2 - x1, y2 - y1]) * scale)
    
    filter_img = cv2.resize(filter_img, (filter_width, int(filter_width * filter_img.shape[0] / filter_img.shape[1])))
    x = int((x1 + x2) / 2) - filter_img.shape[1] // 2
    y = int((y1 + y2) / 2) - filter_img.shape[0] // 2
    
    angle = -np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
    M = cv2.getRotationMatrix2D((filter_img.shape[1] // 2, filter_img.shape[0] // 2), angle, 1)
    rotated_filter = cv2.warpAffine(filter_img, M, (filter_img.shape[1], filter_img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    
    for i in range(rotated_filter.shape[0]):
        for j in range(rotated_filter.shape[1]):
            if rotated_filter[i, j][3] != 0:
                frame[y + i, x + j] = rotated_filter[i, j][:3]

    return frame

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            if filter_option == "Sunglasses":
                overlay_filter(frame, sunglasses, face_landmarks.landmark, [33, 263], 2.3)

    img.image(frame, channels="BGR")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()