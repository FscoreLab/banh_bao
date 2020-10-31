import argparse
import os
import os.path as osp

import cv2
import numpy as np
import pandas as pd
import surface_distance
import tqdm
from scipy.ndimage.measurements import label
from scipy.spatial.distance import directed_hausdorff

from bao.config import system_config
from bao.metrics.ssim import msssim, ssim


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

    tp = len(np.unique(labeled[np.bitwise_and(labeled > 0, img_model > 0)]))
    recall = np.nan if ncomponents == 0 else tp / ncomponents
    precision = np.nan if ncomponents_model == 0 else tp / ncomponents_model
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = (2 * precision * recall) / (precision + recall)

    tmp = {
        "ncomponents": ncomponents,
        "ncomponents_model": ncomponents_model,
        "ncomponents_diff": ncomponents - ncomponents_model,
        "ncomponents_abs_diff": np.abs(ncomponents - ncomponents_model),
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }
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
    tmp = {"ssim": ssim(img_expert, img_model).mean(), "msssim": msssim(img_expert, img_model)}
    return tmp


def surface_distances(img_expert, img_model):
    surface_distances = surface_distance.compute_surface_distances(img_expert, img_model, (0.1, 0.1))
    dist, dist_inv = surface_distance.compute_average_surface_distance(surface_distances)
    robust_hausdorff = surface_distance.compute_robust_hausdorff(surface_distances, 95)
    dice_at_tolerance = surface_distance.compute_surface_dice_at_tolerance(surface_distances, tolerance_mm=1.0)

    tmp = {"dist": dist, "dist": dist_inv, "robust_hausdorff": robust_hausdorff, "dice_at_tolerance": dice_at_tolerance}
    return tmp


def area_features(img_expert, img_model):
    area_expert = img_expert.sum()
    area_model = img_model.sum()
    tmp = {"area_abs_diff": np.abs(area_expert - area_model), "area_expert": area_expert, "area_model": area_model}
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


def calc_metrics(data_expert, data_nn, gt='expert'):
    tmp = {"gt": gt}
    for metric in [
        "inter_over_metrics",
        "binary_feature",
        "hausdorff_distance",
        "ssims",
        "accuracy_features",
        "surface_distances",
        "area_features",
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
            tmp.update(eval(metric)(data_expert, data_nn))
    return tmp


def get_metrics(data, markup=None):
    """
    Arguments
    ---------

    data    (list) : list of dicts {
                    "fname": str, 
                    "orig": RGB 3-channel image,
                    "expert", "m_1", "m_2", "m_3": 2D boolean arrays
                    }
    """

    out_data = []
    sample_name_dict = {"s1": "1", "s2": "2", "s3": "3"}
    for data_dict in tqdm.tqdm(data, desc="Generating metrics"):
        for s_key in ["s1", "s2", "s3"]:
            tmp = {
                "id": f"{data_dict['fname']}_{sample_name_dict[s_key]}",
                "fname": data_dict["fname"],
                "sample_name": sample_name_dict[s_key],
            }

            # Check similarity of scores and generate new features if exist model having score 5
            # If there are several models with score 5, their intersection is not removed now
            if not isinstance(markup, type(None)) and tmp["id"] in markup['id'].values:
                index = pd.Index(markup["id"]).get_loc(tmp["id"])
                if markup["y"][index] == 5:
                    interest_samples = ["s1", "s2", "s3"]
                    interest_samples.remove(s_key)
                    for interest_s_key in interest_samples:
                        tmp.update(calc_metrics(data_dict[s_key], data_dict[interest_s_key], gt=tmp["sample_name"]))
                        out_data.append(tmp)
                    continue

            tmp.update(calc_metrics(data_dict["expert"], data_dict[s_key]))
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

    parser.add_argument("--output_dir", default=osp.join(system_config.data_dir, "interim"))

    args = parser.parse_args()

    output_file = osp.join(args.output_dir, f"{args.task_name}.csv")

    data = read_files(args)

    if args.add_markup:
        markup = prepare_markup(args.markup)
        metrics = get_metrics(data, markup)
    else:
        metrics = get_metrics(data)

    os.makedirs(args.output_dir, exist_ok=True)
    metrics.to_csv(output_file, index=False)
