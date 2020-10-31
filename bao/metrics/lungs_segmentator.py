import lungs_finder as lf
import numpy as np


def lungs_finder_segmentator(img, is_union=True, min_area=int(1024 * 1024 * 0.05)):
    mask = np.zeros((1024, 1024), dtype=int)
    # Try hog
    right = lf.find_right_lung_hog(img)
    left = lf.find_left_lung_hog(img)

    # If hog don't find lungs then try lbp method
    if right is None:
        right = lf.find_right_lung_lbp(img)
        left = lf.find_left_lung_lbp(img)
    # If lbp don't find lungs then try haar method
    if right is None:
        right = lf.find_right_lung_haar(img)
        left = lf.find_left_lung_haar(img)

    # If don't find then return mask filled mask
    if (right is None) or (left is None):
        return mask + 255

    x_r, y_r, width_r, height_r = right
    x_l, y_l, width_l, height_l = left

    # If found lungs is very small
    if (width_r * height_r < min_area) or (width_l * height_l < min_area):
        return mask + 255

    if is_union:
        x_l_union = min(x_r, x_l)
        x_r_union = max(x_r + width_r, x_l + width_l)
        y_u_union = max(y_r + height_r, y_l + height_l)
        y_d_union = min(y_r, y_l)
        mask[y_d_union:y_u_union, x_l_union:x_r_union] = 255
    else:
        mask[y_r:y_r + height_r, x_r:x_r + width_r] = 255
        mask[y_l:y_l + height_l, x_l:x_l + width_l] = 255

    return mask


def area_out_of(lungs_mask, img_model):
    diff_mask = np.bitwise_xor(lungs_mask, img_model)
    out_of_lungs_mask = np.bitwise_and(diff_mask, img_model)

    area_model = int(img_model.sum() / 255.0)  # in pixels
    area_out_of_lungs_mask = int(out_of_lungs_mask.sum() / 255.0)  # in pixels

    if area_model > 0:
        out_of_lungs = area_out_of_lungs_mask / area_model
    else:
        out_of_lungs = 0

    return out_of_lungs
