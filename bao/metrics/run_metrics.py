import argparse
import os
import os.path as osp

import cv2
import numpy as np
import pandas as pd
import tqdm

from bao.config import system_config


def intersection_and_union(img1, img2):
    """
    Arguments
    ---------

    img1, img2  (np.ndarray) : Boolean np.ndarrays
    """
    intersection = (img1 & img2)
    union = (img1 | img2)
    return intersection, union

def iou(intersection, union):
    return np.sum(intersection) / np.sum(union)

def iomin(intersection, img1, img2):
    area_min = min(img1.sum(), img2.sum())
    return np.sum(intersection) / area_min

def iomax(intersection, img1, img2):
    area_max = max(img1.sum(), img2.sum())
    return np.sum(intersection) / area_max

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
        "iomax": iomax(intersection, img1, img2)
    }
    return tmp

def binary_feature(img_expert, img_model):
    """
    Arguments
    ---------

    img_expert, img_model  (np.ndarray) : Boolean np.ndarrays
    """
    gt = (np.sum(img_expert) > 0)
    pred = (np.sum(img_model) > 0)
    tmp = {
        "true": gt == pred,
        "positive_gt": gt,
    }
    return tmp


def _read_png(fpath):
    return cv2.imread(fpath)[:,:,::-1]

def _read_mask(fpath):
    return cv2.imread(fpath).astype(np.bool)

def read_files(args):
    fnames = [osp.splitext(fpath)[0] for fpath in os.listdir(args.folder_origin)]
    data = []
    for fname in fnames:
        data.append({
            "fname": fname,
            "orig": _read_png(osp.join(args.folder_origin, f"{fname}.png")),
            "expert": _read_mask(osp.join(args.folder_expert, f"{fname}_expert.png")),
            "s1": _read_mask(osp.join(args.folder_1, f"{fname}_s1.png")),
            "s2": _read_mask(osp.join(args.folder_2, f"{fname}_s2.png")),
            "s3": _read_mask(osp.join(args.folder_3, f"{fname}_s3.png"))
        })
    return data

def prepare_markup(fpath):
    mark = pd.read_csv(fpath)
    markup = pd.wide_to_long(mark, stubnames="Sample ", i="Case", j="sample_name").reset_index()
    markup = markup.rename(columns={"Sample ": "y"})
    markup["sample_name"] = markup["sample_name"].astype(str)
    markup["fname"] = markup["Case"].map(lambda x: osp.splitext(x)[0])
    markup["id"] = markup[["fname", "sample_name"]].agg('_'.join, axis=1)
    return markup[["id", "y"]]


def get_metrics(data):
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
                "sample_name": sample_name_dict[s_key]
            }
            for metric in ["inter_over_metrics", "binary_feature"]:
                if metric in ["inter_over_metrics", "binary_feature"]:
                    tmp.update(eval(metric)(data_dict["expert"], data_dict[s_key]))
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
    metrics = get_metrics(data)

    if args.add_markup:
        markup = prepare_markup(args.markup)
        metrics = pd.merge(metrics, markup, how="left", on="id")

    os.makedirs(args.output_dir, exist_ok=True)
    metrics.to_csv(output_file, index=False)
