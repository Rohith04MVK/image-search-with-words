import os

import numpy as np
import streamlit as st

from src.captions import get_model
from src.config import IMAGE_FOLDER
from src.db_handler import get_handler
from src.utils import (add_custom_css, display_gallery, find_threshold,
                       load_images, save_uploaded_image)

# Define the folder to store images
image_folder = IMAGE_FOLDER
model = get_model()
db_handler = get_handler()

# Create the folder if it doesn't exist
if not os.path.exists(image_folder):
    os.makedirs(image_folder)


def main():
    add_custom_css()

    st.markdown('<h1 class="main-title">Image Gallery</h1>',
                unsafe_allow_html=True)
    st.markdown('<h2 class="subheader">Capture, Store, and Search Your Favorite Images Effortlessly</h2>',
                unsafe_allow_html=True)
    st.markdown(
        """
        <p class="instructions">
        Semantic search for any(!) image!
        </p>
        """,
        unsafe_allow_html=True
    )

    # Search bar
    search_query = st.text_input("Search images")

    # Upload button
    uploaded_files = st.file_uploader("Choose an image...", type=[
        "jpg", "jpeg", "png", "gif"], accept_multiple_files=True)
    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            img_path = save_uploaded_image(uploaded_file)
            caption = model.generate_caption(img_path)
            db_handler.save_to_chroma_db(caption, img_path)
            st.markdown(
                '<p class="success-message">Image saved successfully!</p>', unsafe_allow_html=True)

    # Load images
    image_files = load_images()

    # Check if there are no images
    if len(image_files) == 0:
        st.write("No images added yet.")
    else:
        # Filter images based on search query
        if search_query:
            result = db_handler.query_chroma_db(search_query)
            threshold = find_threshold(set(result['distances'][0]))

            image_files = [metadata['image_path'] for metadata, distance in zip(
                result['metadatas'][0], result['distances'][0]) if distance <= threshold]
            image_files = list(set(image_files))  # Removing duplicates

        display_gallery(image_files)


if __name__ == "__main__":
    main()
