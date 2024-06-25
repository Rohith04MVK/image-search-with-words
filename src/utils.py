import os

import streamlit as st
from PIL import Image, ImageOps
import numpy as np

from .config import IMAGE_FOLDER


# Function to load images from the directory
def load_images():
    image_files = [os.path.join(IMAGE_FOLDER, file) for file in os.listdir(
        IMAGE_FOLDER) if file.endswith(('jpg', 'jpeg', 'png', 'gif'))]
    return image_files


# Function to display images in a grid
def display_gallery(image_files, columns=3):
    rows = len(image_files) // columns + 1
    for i in range(rows):
        cols = st.columns(columns)
        for j in range(columns):
            index = i * columns + j
            if index < len(image_files):
                with cols[j]:
                    image = Image.open(image_files[index])
                    # Crop and resize to square
                    image = ImageOps.fit(
                        image, (300, 300), Image.Resampling.LANCZOS)
                    st.image(image, use_column_width=True)


# Function to save uploaded image to the folder
def save_uploaded_image(uploaded_file):
    with open(os.path.join(IMAGE_FOLDER, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Saved file : {uploaded_file.name} in {IMAGE_FOLDER}")
    return os.path.join(IMAGE_FOLDER, uploaded_file.name)


def find_threshold(distances, min_samples=3, jump_factor=1.5):
    if len(distances) < 2:
        return max(distances) if distances else None

    sorted_distances = sorted(distances)

    if len(sorted_distances) < min_samples:
        # For very few samples, use a high percentile
        return sorted_distances[int(len(sorted_distances) * 0.75)]

    for i in range(1, len(sorted_distances)):
        if sorted_distances[i] > sorted_distances[i-1] * jump_factor:
            # Found a significant jump, set threshold between the two values
            return (sorted_distances[i] + sorted_distances[i-1]) / 2

    # If no significant jump is found, return a high percentile
    return sorted_distances[int(len(sorted_distances) * 0.75)]


def add_custom_css():
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 2.5em;
            color: #4A90E2;
            text-align: center;
            margin-top: 20px;
        }
        .subheader {
            font-size: 1.5em;
            color: #7B8DAB;
            text-align: center;
            margin-bottom: 20px;
        }
        .instructions {
            font-size: 1.2em;
            color: #6C757D;
            text-align: center;
            margin-bottom: 30px;
        }
        .success-message {
            color: #6FC276;
            font-weight: bold;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
