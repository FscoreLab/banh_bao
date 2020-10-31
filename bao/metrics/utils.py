import lungs_finder as lf
import numpy as np
import cv2


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


def get_centers_of_mass(image):
    # find contours in the binary image
    contours = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cxs = []
    cys = []
    for c in contours[0]:
        # Calculate moments for each contour
        M = cv2.moments(c)

        m00 = M["m00"]
        if m00 == 0:
            m00 = 1
            
        # Calculate x,y coordinate of center
        cX = int(M["m10"] / m00)
        cY = int(M["m01"] / m00)

        cxs.append(cX)
        cys.append(cY)

    return (cxs, cys)


def get_center_of_mass(image):
    cxs, cys = get_centers_of_mass(image)
    cx_mass = np.mean(cxs)
    cy_mass = np.mean(cys)
    return (cx_mass, cy_mass)


def get_nearest_neighbor_dist(points_lhs, points_rhs):
    points_lhs = [(x, y) for (x, y) in zip(points_lhs[0], points_lhs[1])]
    points_rhs = [(x, y) for (x, y) in zip(points_rhs[0], points_rhs[1])]

    if len(points_lhs) > len(points_rhs):
        points_lhs, points_rhs = points_rhs, points_lhs

    min_values = []
    min_value = 5000
    min_p = []
    while not len(points_lhs) == 0 and not len(points_rhs) == 0:
        for i, p in enumerate(points_rhs):
            dist = np.linalg.norm((points_lhs[0][0]-p[0], points_lhs[0][1]-p[1]))

            if dist < min_value:
                min_value = dist
                min_p = p
        min_values.append(min_value)
        points_lhs.remove(points_lhs[0])
        points_rhs.remove(min_p)

        min_value = 5000
        min_ind = 0
    return min_values


def get_lungs_size(img, min_area=int(1024 * 1024 * 0.05)):
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
        return 1024, 1024

    x_r, y_r, width_r, height_r = right
    x_l, y_l, width_l, height_l = left

    # If found lungs is very small
    if (width_r * height_r < min_area) or (width_l * height_l < min_area):
        return 1024, 1024

    x_l_union = min(x_r, x_l)
    x_r_union = max(x_r + width_r, x_l + width_l)
    y_u_union = max(y_r + height_r, y_l + height_l)
    y_d_union = min(y_r, y_l)

    w = x_r_union - x_l_union
    h = y_u_union - y_d_union
    return w, h
