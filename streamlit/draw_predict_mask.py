import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

import streamlit as st
import streamlit.components.v1 as components
import shap
from bao.train_model import CustomRegressor
from bao.inference.predict import predict


def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# Specify canvas parameters in application
stroke_width = 3
stroke_color = "black"
bg_color = "#eee"
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
gt_mask_file = st.sidebar.file_uploader("Ground truth mask:", type=["png", "jpg"])
drawing_mode = st.sidebar.selectbox("Drawing tool:", ("rect", "transform"))
realtime_update = st.sidebar.checkbox("Update in realtime", True)

image = None
if bg_image and gt_mask_file:
    gt_mask = np.array(Image.open(gt_mask_file))
    orig_image = np.array(Image.open(bg_image))
    image = np.uint8(np.stack((orig_image,) * 3, axis=-1))
    mask_red = np.zeros(image.shape)
    mask_red[:, :] = (255, 0, 0)
    mask_red = np.uint8(mask_red * gt_mask[:, :, np.newaxis])
    image[mask_red > 0] = image[mask_red > 0] * 0.5 + mask_red[mask_red > 0] * 0.5
    image = Image.fromarray(image)

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color="",
    background_image=image,
    update_streamlit=realtime_update,
    height=512,
    width=512,
    drawing_mode=drawing_mode,
    key="canvas",
)

if canvas_result.json_data is not None and bg_image is not None and gt_mask_file is not None:
    results = canvas_result.json_data["objects"]
    mask = np.zeros((1024, 1024), dtype=bool)
    for res in results:

        mask[int(res["top"]) * 2 : (int(res["top"]) + int(res["height"])) * 2, int(res["left"]) * 2 : (int(res["left"]) + int(res["width"])) * 2] = 1

    prediction, shap_obj = predict(orig_image, gt_mask.astype(bool), mask, return_shap=True)
    st.markdown(f"# Prediction: {prediction}")
    st_shap(shap_obj)
