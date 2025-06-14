import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="VIBGYOR Color Detection", layout="centered")
st.title("🌈 VIBGYOR Color Detection App")

st.markdown("Upload an image to detect Violet, Indigo, Blue, Green, Yellow, Orange, and Red regions.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    height, width = image_cv.shape[:2]
    if width > 800:
        scale = 800 / width
        image_cv = cv2.resize(image_cv, (int(width * scale), int(height * scale)))

    hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
    output = image_cv.copy()

    color_ranges = {
        'Violet':  ([125, 50, 70],  [155, 255, 255]),
        'Indigo':  ([110, 50, 70],  [124, 255, 255]),
        'Blue':    ([90, 50, 70],   [109, 255, 255]),
        'Green':   ([40, 50, 70],   [85, 255, 255]),
        'Yellow':  ([25, 50, 70],   [35, 255, 255]),
        'Orange':  ([10, 100, 20],  [24, 255, 255]),
        'Red':     ([0, 120, 70],   [10, 255, 255]),
        'Crimson': ([170, 120, 70], [180, 255, 255])
    }

    label_colors = {
        'Violet':  (211, 0, 148),
        'Indigo':  (130, 0, 75),
        'Blue':    (255, 0, 0),
        'Green':   (0, 255, 0),
        'Yellow':  (0, 255, 255),
        'Orange':  (0, 165, 255),
        'Red':     (0, 0, 255),
        'Crimson': (60, 20, 220)
    }

    min_area = 800  

    for color in color_ranges:
        lower = np.array(color_ranges[color][0])
        upper = np.array(color_ranges[color][1])
        mask = cv2.inRange(hsv, lower, upper)

        if color == 'Red':
            lower2 = np.array(color_ranges['Crimson'][0])
            upper2 = np.array(color_ranges['Crimson'][1])
            mask += cv2.inRange(hsv, lower2, upper2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(output, (x, y), (x + w, y + h), label_colors[color], 2)
                cv2.putText(output, color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_colors[color], 2)

    result_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    st.image(result_rgb, caption="Detected Colors", use_column_width=True)

else:
    st.info("Please upload an image to begin.")
