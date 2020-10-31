import argparse
import os
import os.path as osp
import glob

import numpy as np
import torch
import pandas as pd
import tqdm
import torchxrayvision as xrv

from bao.config import system_config
from bao.torch_config import torch_config
from bao.metrics.run_metrics import _read_png


def load_model():
    model = xrv.models.DenseNet(weights="all")
    model.to(torch_config.device)
    model.eval()

    return model


def gather_file_info(folder_origin):
    fpaths = glob.glob(f"{folder_origin}/*")
    img_dict = {osp.basename(fpath).split(".")[0]: fpath for fpath in fpaths}
    data = []
    for fname, img_path in img_dict.items():
        data.append({
            "orig": img_path,
        })

    return data


def _make_1d_image(img):
    # Make image 2D array
    if len(img.shape) > 2:
        img = img[:, :, 0]
    assert len(img.shape) >= 2, f"Image should contain at least 12 dimensions, got shape {img.shape}"
    return img


def _prepare_image(image_3channels):
    img = xrv.datasets.normalize(image_3channels, 255)
    img = _make_1d_image(img)

    # Add color channel
    img = img[None, :, :]  
    return img


def _get_probs(model, image_3channels, transform=torch_config.transform):
    """
    Arguments
    ---------
    model      (torch.nn.Module) : Model for inference
    image_3channels (np.ndarray) : Image in format (H, W, C) with 3 channels
    """
    img = _prepare_image(image_3channels)
    img = transform(img)
    image_tensor = torch.from_numpy(img).unsqueeze(0)
    image_tensor = image_tensor.to(torch_config.device)
    with torch.no_grad():
        res = model(image_tensor)
        probs_NIH = res.detach().cpu().numpy()
    return probs_NIH.squeeze()

def get_probs_for_file(model, image_fname):
    image_3channels = _read_png(image_fname)
    probs = _get_probs(model, image_3channels)

    classes = model.pathologies
    case_name = osp.basename(image_fname)
    df = pd.DataFrame.from_dict({case_name: probs}, orient="index", columns=classes)

    df["No Finding"] = 1 - df.max(axis=1)
    df["No Finding Sum"] = -df.sum(axis=1)
    df["fname"] = df.index.str.split(".").str[0]

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate predictions of XRay model")

    parser.add_argument("--task_name", default="model_torchxray")
    parser.add_argument("--folder_origin", default=osp.join(system_config.data_dir, "Dataset", "Origin"))
    parser.add_argument("--output_dir", default=osp.join(system_config.data_dir, "interim"))
    
    args = parser.parse_args()

    np.random.seed(24)
    torch.manual_seed(24)
    torch.cuda.manual_seed(24)

    model = load_model()
    file_info = gather_file_info(args.folder_origin)
    preds_final = []
    for file_pair in tqdm.tqdm(file_info, "Get network predictions"):
        preds_final.append(get_probs_for_file(model, file_pair["orig"]))
    df = pd.concat(preds_final)

    os.makedirs(args.folder_origin, exist_ok=True)
    df.to_csv(osp.join(args.output_dir, f"{args.task_name}.csv"))
