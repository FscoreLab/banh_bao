import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

import streamlit as st
from bao.train_model import CustomRegressor
from bao.inference.predict import predict


# Specify canvas parameters in application
stroke_width = 3
stroke_color = "black"
bg_color = "#eee"
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
gt_mask_file = st.sidebar.file_uploader("Ground truth mask:", type=["png", "jpg"])
pred_mask_file = st.sidebar.file_uploader("Predicted mask:", type=["png", "jpg"])
realtime_update = st.sidebar.checkbox("Update in realtime", True)

image = None
if bg_image:
    orig_image = np.array(Image.open(bg_image))
    image = np.uint8(np.stack((orig_image,) * 3, axis=-1))
    image = Image.fromarray(image)

if image and gt_mask_file:
    gt_mask = np.array(Image.open(gt_mask_file))
    image = np.asarray(image).copy()
    mask_red = np.zeros(image.shape)
    mask_red[:, :] = (255, 0, 0)
    mask_red = np.uint8(mask_red * gt_mask[:, :, np.newaxis])
    image[mask_red > 0] = image[mask_red > 0] * 0.5 + mask_red[mask_red > 0] * 0.5
    image = Image.fromarray(image)

if image and pred_mask_file:
    pred_mask = np.array(Image.open(pred_mask_file))
    image = np.asarray(image).copy()
    mask_green = np.zeros(image.shape)
    mask_green[:, :] = (0, 255, 0)
    mask_green = np.uint8(mask_green * pred_mask[:, :, np.newaxis])
    image[mask_green > 0] = image[mask_green > 0] * 0.5 + mask_red[mask_green > 0] * 0.5
    image = Image.fromarray(image)

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color="",
    background_image=image,
    update_streamlit=realtime_update,
    height=512,
    width=512,
    key="canvas",
)

if bg_image is not None and gt_mask_file is not None and pred_mask_file is not None:
    prediction = predict(orig_image, gt_mask.astype(bool), pred_mask.astype(bool))

    st.markdown(f"# Prediction: {prediction}")
