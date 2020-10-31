import cv2
import numpy as np


def get_objects(img):
    """
    Arguments
    ---------
    img     (np.ndarray) : Boolean array

    Returns
    -------
    objs     (list of dicts) : One dict has structure {
                                "rect": [(x_top_left, y_top_left), (x_bottom_right, y_bottom_right)]
                                "el": [(x_center, y_center), (x_leght, y_length)]
                                }
    """

    countours, _ = cv2.findContours(img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    objs = []
    for countour in countours:
        x_min = np.min(countour[:, :, 0])
        x_max = np.max(countour[:, :, 0])
        y_min = np.min(countour[:, :, 1])
        y_max = np.max(countour[:, :, 1])
        x_c = (x_min + x_max) // 2
        y_c = (y_min + y_max) // 2
        x_len = (x_max - x_min) // 2
        y_len = (y_max - y_min) // 2
        objs.append({"rect": [(x_min, y_min), (x_max, y_max)], "el": [(x_c, y_c), (x_len, y_len)]})
    return objs


def draw_rectangles(objs, shape):
    """
    Arguments
    ---------
    objs     (list of dicts) : One dict has structure {
                                "rect": [(x_top_left, y_top_left), (x_bottom_right, y_bottom_right)]
                                "el": [(x_center, y_center), (x_leght, y_length)]
                                }
    shape            (tuple) : shape of the image

    Returns
    -------
    img         (np.ndarray) : Array of booleans with only rectangles
    """
    img = np.zeros(shape, dtype=np.uint8)
    for obj in objs:
        cv2.rectangle(img, obj["rect"][0], obj["rect"][1], 1, -1)
    img = img.astype(np.bool)
    return img


def draw_ellipses(objs, shape):
    """
    Arguments
    ---------
    objs     (list of dicts) : One dict has structure {
                                "rect": [(x_top_left, y_top_left), (x_bottom_right, y_bottom_right)]
                                "el": [(x_center, y_center), (x_leght, y_length)]
                                }
    shape            (tuple) : shape of the image

    Returns
    -------
    img         (np.ndarray) : Array of booleans with only ellipses
    """
    img = np.zeros(shape, dtype=np.uint8)
    for obj in objs:
        cv2.ellipse(img, obj["el"][0], obj["el"][1], 0, 0, 360, 1, -1)
    img = img.astype(np.bool)
    return img
