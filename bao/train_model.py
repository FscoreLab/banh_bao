import argparse
import os
import os.path as osp
import pickle
import random

import lightgbm
import numpy as np
import pandas as pd
import shap
from sklearn import metrics
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    cross_val_predict,
)

from bao.config import system_config


# could not use Pipeline, sklearn doesn't allow cloning Pipelines
class CustomRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, **estimator_params):
        super().__init__()
        self.selector = SelectKBest(score_func=mutual_info_regression, k="all")
        self.base_model = lightgbm.LGBMRegressor(random_state=24, objective="regression_l1")
        self.set_params(**estimator_params)

    def fit(self, X, y=None):
        X_tr = self.selector.fit_transform(X, y)
        self.base_model.fit(X_tr, y)
        return self

    def predict(self, X):
        X_tr = self.selector.transform(X)
        y = self.base_model.predict(X_tr)
        y[X.false_positive == 1] = 1
        y[X.false_negative == 1] = 1
        y[(X.area_model + X.area_expert) == 0] = 5
        y = np.round(y, 0)
        return y

    def get_params(self, **params):
        pars = self.base_model.get_params()
        pars["k"] = self.selector.k
        return pars

    def set_params(self, **params):
        if "k" in params:
            k = params.pop("k")
            self.selector = self.selector.set_params(k=k)
        self.base_model = self.base_model.set_params(**params)
        return self


def preprocess_features(df):
    df["false_positive"] = ~df.true & ~df.positive_gt
    df["false_negative"] = ~df.true & df.positive_gt
    df[["iou", "iomax", "dice_at_tolerance"]] = df[["iou", "iomax", "dice_at_tolerance"]].fillna(1.0)
    df.loc[~df.positive_gt & df.true, "iomin"] = df.loc[~df.positive_gt & df.true, "iomin"].fillna(1.0)
    df["iomin"].fillna(0.0, inplace=True)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model")

    parser.add_argument("--inner_splits", type=int, default=10, help="number of inner folds")
    parser.add_argument("--outer_splits", type=int, default=10, help="number of outer folds")
    parser.add_argument("--plot_shap", action="store_true")
    parser.add_argument("--evaluate_with_nested_cv", action="store_true")

    args = parser.parse_args()

    np.random.seed(24)
    random.seed(24)

    df = pd.read_csv(osp.join(system_config.data_dir, "interim", "sample.csv"))
    df = df.loc[df["gt"] == "expert"]
    df_torch = pd.read_csv(osp.join(system_config.data_dir, "interim", "model_torchxray.csv"))
    df_torch.drop(["gt_sum", "gt_sum_pos"], axis=1, inplace=True, errors="ignore")

    df = pd.merge(df, df_torch, on="fname")
    df = preprocess_features(df)

    predictors = [col for col in df.columns if col not in ["id", "fname", "sample_name", "y", "Unnamed: 0", "gt"]]

    df[predictors] = df[predictors].fillna(0.0)
    df[predictors] = df[predictors].replace([np.inf], 100000.0)

    df_train = df.loc[~pd.isnull(df.y)].copy()
    df_test = df = df.loc[pd.isnull(df.y)].copy()

    X_train = df_train[predictors].copy()
    y_train = df_train["y"]

    inner_cv = GroupKFold(n_splits=args.inner_splits)
    outer_cv = GroupKFold(n_splits=args.outer_splits)

    model = CustomRegressor()
    param_grid = {
        "n_estimators": [50, 100, 200, 300],
        "colsample_bytree": [0.25, 0.5, 0.75, 1.0],
        "min_child_samples": [1, 3, 5, 10],
        "num_leaves": [5, 7, 15, 31],
        "k": [15, 30, len(predictors)],
        "reg_alpha": [0.0, 0.1, 1.0],
        "reg_lambda": [0.0, 0.1, 1.0],
    }

    nested_score = None
    nested_accuracy = None
    if args.evaluate_with_nested_cv:
        pred_y = []
        true_y = []
        for train_index, test_index in outer_cv.split(X_train, y_train, groups=df_train.fname):
            X_tr, X_tt = X_train.iloc[train_index, :], X_train.iloc[test_index, :]
            y_tr, y_tt = y_train.iloc[train_index], y_train.iloc[test_index]
            groups_tt = df_train.fname.iloc[train_index]

            clf = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=inner_cv,
                verbose=True,
                n_jobs=-1,
                scoring="neg_mean_absolute_error",
            )
            clf.fit(X_tr, y_tr, groups=groups_tt)

            pred = clf.predict(X_tt)
            pred_y.extend(pred)
            true_y.extend(y_tt)
            nested_score = metrics.mean_absolute_error(true_y, pred_y)
            nested_accuracy = metrics.accuracy_score(true_y, pred_y)
            print(nested_score, nested_accuracy)

        nested_score = metrics.mean_absolute_error(true_y, pred_y)
        nested_accuracy = metrics.accuracy_score(true_y, pred_y)

    clf = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
        verbose=True,
        scoring="neg_mean_absolute_error",
    )
    clf.fit(X_train, y_train, groups=df_train.fname)
    non_nested_score = -clf.best_score_

    print(
        f"Final metrics: Nested L1 {nested_score}, Nested Accuracy {nested_accuracy}, Non-Nested L1{non_nested_score}"
    )

    if args.plot_shap:
        X_tr = clf.best_estimator_.selector.transform(X_train)
        explainer = shap.TreeExplainer(clf.best_estimator_.base_model)
        shap_values = explainer.shap_values(X_tr)
        topk = clf.best_estimator_.selector.get_support()
        shap.summary_plot(shap_values, X_tr, axis_color="white", feature_names=X_train.columns[topk])

    model = model.set_params(**clf.best_params_)
    df_train["prediction"] = cross_val_predict(model, X_train, y_train, cv=inner_cv, groups=df_train.fname)
    df_test["prediction"] = clf.predict(df_test[predictors])

    os.makedirs(osp.join(system_config.data_dir, "processed"), exist_ok=True)

    df_train[["id", "prediction", "y"]].to_csv(
        osp.join(system_config.data_dir, "processed", "train_predictions.csv"), index=False
    )
    df_test[["id", "prediction"]].to_csv(
        osp.join(system_config.data_dir, "processed", "test_predictions.csv"), index=False
    )

    with open(osp.join(system_config.data_dir, "processed", "model.pkl"), "wb") as fout:
        pickle.dump(clf.best_estimator_, fout)
