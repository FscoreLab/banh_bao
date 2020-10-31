import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# Specify canvas parameters in application
stroke_width = 3
stroke_color = "black"
bg_color = "#eee"
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
gt_mask = st.sidebar.file_uploader("Ground truth mask:", type=["png", "jpg"])
drawing_mode = st.sidebar.selectbox("Drawing tool:", ("rect", "transform"))
realtime_update = st.sidebar.checkbox("Update in realtime", True)

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color="" if bg_image else bg_color,
    background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    height=512,
    width=512,
    drawing_mode=drawing_mode,
    key="canvas",
)

if canvas_result.json_data is not None and bg_image is not None and gt_mask is not None:
    # Insert prediction generation here
    results = canvas_result.json_data["objects"]
    mask = np.zeros((1024, 1024), dtype=bool)
    for res in results:
        mask[res["top"] * 2 : (res["top"] + res["height"]) * 2, res["left"] * 2 : (res["left"] + res["width"]) * 2] = 1

    gt_mask = np.array(Image.open(gt_mask))
    orig_image = np.array(bg_image)

    # predict here
