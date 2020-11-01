import argparse
import os
import os.path as osp
import pickle

import numpy as np
import pandas as pd
import shap

from bao.config import system_config, api_config
from bao.metrics.run_net import load_model as load_neural_network
from bao.metrics.run_net import get_probs_for_3ch_image
from bao.metrics import mask_utils
from bao.metrics.run_metrics import calc_metrics
from bao.train_model import preprocess_features, CustomRegressor
from bao.metrics.run_metrics import _read_png, _read_mask


def load_regressor():
    model_path = osp.join(system_config.model_dir, api_config.model_fname)
    with open(model_path, "rb") as fin:
        model = pickle.load(fin)
    return model


def create_form_dict(img_expert, img_model):
    form_dict = {}
    form_dict["expert_ellipse"] = mask_utils.convert_to_ellipses(img_expert)
    form_dict["expert_rect"] = mask_utils.convert_to_rectangles(img_expert)
    form_dict["model_ellipse"] = mask_utils.convert_to_ellipses(img_model)
    form_dict["model_rect"] = mask_utils.convert_to_rectangles(img_model)

    return form_dict


def predict(img_origin, img_expert, img_model, return_shap=False):
    # Get prediction for exernal model
    df_torch = get_probs_for_3ch_image(NEURAL_NET, img_origin)

    # Get prediction for metrics
    form_dict = create_form_dict(img_expert, img_model)
    tmp = {"fname": "sample"}
    tmp.update(calc_metrics(img_expert, img_model, img_origin, form_dict))
    df = pd.DataFrame([tmp])

    # Unite predictors
    df = pd.merge(df, df_torch, on="fname")

    # Preprocessing
    df = preprocess_features(df)
    predictors = [col for col in df.columns if col not in ["id", "fname", "sample_name", "y", "Unnamed: 0", "gt"]]

    df[predictors] = df[predictors].fillna(0.0)
    df[predictors] = df[predictors].replace([np.inf], 100000.0)

    # Prediction
    prediction = MODEL.predict(df[predictors])[0]

    if return_shap:
        X_tr = MODEL.selector.transform(df[predictors])
        explainer = shap.TreeExplainer(MODEL.base_model)
        shap_values = explainer.shap_values(X_tr)
        topk = MODEL.selector.get_support()
        shap_obj = shap.force_plot(
            explainer.expected_value, shap_values[0, :], X_tr[0, :], feature_names=df[predictors].columns[topk]
        )
        return prediction, shap_obj

    return prediction


NEURAL_NET = load_neural_network()
MODEL = load_regressor()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Get prediction for 1 pair of files")
    parser.add_argument(
        "--file_orig", default=osp.join(system_config.data_dir, "Dataset", "Origin", "00013977_005.png")
    )
    parser.add_argument(
        "--file_expert", default=osp.join(system_config.data_dir, "Dataset", "Expert", "00013977_005_expert.png")
    )
    parser.add_argument(
        "--file_model", default=osp.join(system_config.data_dir, "Dataset", "sample_1", "00013977_005_s1.png")
    )
    parser.add_argument("--output_dir", default=osp.join(system_config.data_dir, "prediction"))

    args = parser.parse_args()

    img_origin = _read_png(args.file_orig)
    img_expert = _read_mask(args.file_expert)
    img_model = _read_mask(args.file_model)

    pred = predict(img_origin, img_expert, img_model)
    print(pred)

    os.makedirs(args.output_dir, exist_ok=True)
    case_name = osp.splitext(osp.basename(args.file_model))[0]
    df = pd.DataFrame([{"id": case_name, "pred": pred}])
    df.to_csv(osp.join(args.output_dir, f"{case_name}.csv"), index=False)
