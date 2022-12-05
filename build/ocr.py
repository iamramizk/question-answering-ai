import pytesseract
from PIL import Image
import streamlit as st


@st.cache
def text_from_images(images: list) -> str:
    corpus = ""
    for image in images:
        img = Image.open(image)
        text = pytesseract.image_to_string(img)
        if text:
            corpus += text
    return corpus
