import argparse
import os
import os.path as osp
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import surface_distance
import tqdm
from scipy.ndimage.measurements import label
from scipy.spatial.distance import directed_hausdorff

from bao.config import system_config
from bao.metrics.ssim import ssim
from bao.metrics.lungs_segmentator import lungs_finder_segmentator, area_out_of
from bao.metrics import mask_utils


def intersection_and_union(img1, img2):
    """
    Arguments
    ---------

    img1, img2  (np.ndarray) : Boolean np.ndarrays
    """
    intersection = img1 & img2
    union = img1 | img2
    return intersection, union


def iou(intersection, union):
    return np.sum(intersection) / np.sum(union)


def iomin(intersection, img1, img2):
    area_min = min(img1.sum(), img2.sum())
    return np.sum(intersection) / area_min


def iomax(intersection, img1, img2):
    area_max = max(img1.sum(), img2.sum())
    return np.sum(intersection) / area_max


def dice(intersection, img1, img2, smooth=1):
    area_sum = img1.sum() + img2.sum()
    return (2 * np.sum(intersection) + smooth) / (area_sum + smooth)


def inter_over_metrics(img1, img2):
    """
    Arguments
    ---------

    img1, img2  (np.ndarray) : Boolean np.ndarrays
    """
    intersection, union = intersection_and_union(img1, img2)
    tmp = {
        "iou": iou(intersection, union),
        "iomin": iomin(intersection, img1, img2),
        "iomax": iomax(intersection, img1, img2),
        "dice": dice(intersection, img1, img2),
    }
    return tmp


def binary_feature(img_expert, img_model):
    """
    Arguments
    ---------

    img_expert, img_model  (np.ndarray) : Boolean np.ndarrays
    """
    gt = np.sum(img_expert) > 0
    pred = np.sum(img_model) > 0
    tmp = {
        "true": gt == pred,
        "positive_gt": gt,
    }
    return tmp


def accuracy_features(img_expert, img_model):
    structure = np.ones((3, 3), dtype=np.int)
    labeled, ncomponents = label(img_expert, structure)
    labeled_model, ncomponents_model = label(img_model, structure)

    tmp = {
        "ncomponents": ncomponents,
        "ncomponents_model": ncomponents_model,
        "ncomponents_abs_diff": np.abs(ncomponents - ncomponents_model),
    }

    correct_pred = labeled[np.bitwise_and(labeled > 0, img_model > 0)]
    intersection = Counter(correct_pred)
    union = Counter(labeled[np.bitwise_or(labeled > 0, img_model > 0)])
    true_labels = np.unique(correct_pred)
    ious = {}
    for obj_label in true_labels:
        ious[obj_label] = intersection.get(obj_label, 0.0) / union.get(obj_label, 0.01)

    iou_thresholds = (0.0, 0.25, 0.5)
    for iou_threshold in iou_thresholds:
        tp = len([obj_label for obj_label in true_labels if ious[obj_label] > iou_threshold])
        recall = 1.0 if ncomponents == 0 else tp / ncomponents
        precision = 1.0 if ncomponents_model == 0 else tp / ncomponents_model
        tmp[f"recall_{iou_threshold}"] = recall
        tmp[f"precision_{iou_threshold}"] = precision
        if precision + recall == 0.0:
            tmp[f"f1_{iou_threshold}"] = 0.0
        else:
            tmp[f"f1_{iou_threshold}"] = (2 * precision * recall) / (precision + recall)

    return tmp


def hausdorff_distance(img_expert, img_model):
    """
    Arguments
    ---------

    img_expert, img_model  (np.ndarray) : Boolean np.ndarrays
    """
    tmp = {
        "hausdorff": directed_hausdorff(img_expert, img_model, seed=24)[0],
        "hausdorff_inv": directed_hausdorff(img_model, img_expert, seed=24)[0],
    }
    return tmp


def ssims(img_expert, img_model):
    """
    Arguments
    ---------

    img_expert, img_model  (np.ndarray) : Boolean np.ndarrays
    """
    tmp = {"ssim": ssim(img_expert, img_model).mean()}
    return tmp


def surface_distances(img_expert, img_model):
    surface_distances = surface_distance.compute_surface_distances(img_expert, img_model, (0.1, 0.1))
    dist, dist_inv = surface_distance.compute_average_surface_distance(surface_distances)
    robust_hausdorff = surface_distance.compute_robust_hausdorff(surface_distances, 95)
    dice_at_tolerance = surface_distance.compute_surface_dice_at_tolerance(surface_distances, tolerance_mm=1.0)

    tmp = {
        "dist": dist,
        "dist_inv": dist_inv,
        "robust_hausdorff": robust_hausdorff,
        "dice_at_tolerance": dice_at_tolerance,
    }
    return tmp


def area_features(img_expert, img_model):
    area_expert = img_expert.sum()
    area_model = img_model.sum()
    tmp = {"area_abs_diff": np.abs(area_expert - area_model), "area_expert": area_expert, "area_model": area_model}
    return tmp


def area_out_of_lungs(img_origin, img_expert, img_model):
    # Calculate part of mask out from lungs
    lungs_mask_union = lungs_finder_segmentator(img_origin)
    out_of_lungs_union = area_out_of(lungs_mask_union, img_model)

    lungs_mask_lr = lungs_finder_segmentator(img_origin, is_union=False)
    out_of_lungs_lr = area_out_of(lungs_mask_lr, img_model)

    tmp = {
        "out_of_lungs_lr": out_of_lungs_lr,
        "out_of_lungs_union": out_of_lungs_union,
    }
    return tmp


def _read_png(fpath):
    return cv2.imread(fpath)[:, :, ::-1]


def _read_mask(fpath):
    mask = cv2.imread(fpath).astype(np.bool)
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    return mask


def read_files(args):
    fnames = [osp.splitext(fpath)[0] for fpath in os.listdir(args.folder_origin)]
    data = []
    for fname in fnames:
        data.append(
            {
                "fname": fname,
                "orig": _read_png(osp.join(args.folder_origin, f"{fname}.png")),
                "expert": _read_mask(osp.join(args.folder_expert, f"{fname}_expert.png")),
                "s1": _read_mask(osp.join(args.folder_1, f"{fname}_s1.png")),
                "s2": _read_mask(osp.join(args.folder_2, f"{fname}_s2.png")),
                "s3": _read_mask(osp.join(args.folder_3, f"{fname}_s3.png")),
            }
        )
    return data


def prepare_markup(fpath):
    mark = pd.read_csv(fpath)
    markup = pd.wide_to_long(mark, stubnames="Sample ", i="Case", j="sample_name").reset_index()
    markup = markup.rename(columns={"Sample ": "y"})
    markup["sample_name"] = markup["sample_name"].astype(str)
    markup["fname"] = markup["Case"].map(lambda x: osp.splitext(x)[0])
    markup["id"] = markup[["fname", "sample_name"]].agg("_".join, axis=1)
    return markup[["id", "y"]]


def _add_key_postfix(dictionary, postfix):
    """
    Adds postfix to every key in dictionary
    """
    key_pairs = {old_key: f"{old_key}{postfix}" for old_key in list(dictionary.keys())}
    new_dict = {}
    for old_key, value in dictionary.items():
        new_dict[key_pairs[old_key]] = value
    return new_dict


def get_metrics(data, form_mode="original"):
    """
    Arguments
    ---------

    data    (list) : list of dicts {
                    "fname": str,
                    "orig": RGB 3-channel image,
                    "expert", "m_1", "m_2", "m_3": 2D boolean arrays
                    }
    form_mode   (str) : If `original`, add features for ellipses and 
                        rectangles for selected metrics, if `rect` - 
                        generate features for rectangle masks, 
                        if `ellipse` - generate features for ellipsoid masks
    """

    out_data = []
    sample_name_dict = {"s1": "1", "s2": "2", "s3": "3"}
    for data_dict in tqdm.tqdm(data, desc="Generating metrics"):
        if form_mode in ["rect", "ellipse"]:
            for markup_key in ["expert", "s1", "s2", "s3"]:
                if form_mode == "rect":
                    data_dict[markup_key] = mask_utils.convert_to_rectangles(data_dict[markup_key])
                elif form_mode == "ellipse":
                    data_dict[markup_key] = mask_utils.convert_to_ellipses(data_dict[markup_key])

        if form_mode == "original":
            expert_ellipse = mask_utils.convert_to_ellipses(data_dict["expert"])
            expert_rect = mask_utils.convert_to_rectangles(data_dict["expert"])

        for s_key in ["s1", "s2", "s3"]:

            tmp = {
                "id": f"{data_dict['fname']}_{sample_name_dict[s_key]}",
                "fname": data_dict["fname"],
                "sample_name": sample_name_dict[s_key],
            }

            if form_mode == "original":
                model_ellipse = mask_utils.convert_to_ellipses(data_dict[s_key])
                model_rect = mask_utils.convert_to_rectangles(data_dict[s_key])

            for metric in [
                "inter_over_metrics",
                "binary_feature",
                "hausdorff_distance",
                "ssims",
                "accuracy_features",
                "surface_distances",
                "area_features",
                "area_out_of_lungs",
            ]:
                if metric in [
                    "inter_over_metrics",
                    "binary_feature",
                    "hausdorff_distance",
                    "ssims",
                    "accuracy_features",
                    "surface_distances",
                    "area_features",
                ]:
                    tmp.update(eval(metric)(data_dict["expert"], data_dict[s_key]))

                if metric in [
                    "area_out_of_lungs"
                ]:
                    tmp.update(eval(metric)(data_dict["orig"], data_dict["expert"], data_dict[s_key]))

                if metric in ["inter_over_metrics", "hausdorff_distance"] and form_mode == "original":
                    tmp_tmp = eval(metric)(expert_ellipse, model_ellipse)
                    tmp_tmp = _add_key_postfix(tmp_tmp, "_el")
                    tmp.update(tmp_tmp)
                    tmp_tmp = eval(metric)(expert_rect, model_rect)
                    tmp_tmp = _add_key_postfix(tmp_tmp, "_rect")
                    tmp.update(tmp_tmp)

            out_data.append(tmp)

    return pd.DataFrame(out_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Gather metrics")

    parser.add_argument("--task_name", default="sample")

    parser.add_argument("--folder_origin", default=osp.join(system_config.data_dir, "Dataset", "Origin"))
    parser.add_argument("--folder_expert", default=osp.join(system_config.data_dir, "Dataset", "Expert"))
    parser.add_argument("--folder_1", default=osp.join(system_config.data_dir, "Dataset", "sample_1"))
    parser.add_argument("--folder_2", default=osp.join(system_config.data_dir, "Dataset", "sample_2"))
    parser.add_argument("--folder_3", default=osp.join(system_config.data_dir, "Dataset", "sample_3"))

    parser.add_argument("--markup", default=osp.join(system_config.data_dir, "Dataset", "OpenPart.csv"))
    parser.add_argument("--add_markup", action="store_true")
    parser.add_argument(
        "--form_mode",
        default="original",
        help="If `original`, add features for ellipses and rectangles for selected "
        + "metrics, if `rect` generate features for rectangle masks, if `ellipse`"
        + " generate features for ellipsoid masks",
    )

    parser.add_argument("--output_dir", default=osp.join(system_config.data_dir, "interim"))

    args = parser.parse_args()

    output_file = osp.join(args.output_dir, f"{args.task_name}.csv")

    data = read_files(args)
    metrics = get_metrics(data, form_mode=args.form_mode)

    if args.add_markup:
        markup = prepare_markup(args.markup)
        metrics = pd.merge(metrics, markup, how="left", on="id")

    os.makedirs(args.output_dir, exist_ok=True)
    metrics.to_csv(output_file, index=False)
