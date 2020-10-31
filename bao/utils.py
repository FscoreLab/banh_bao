import numpy as np
import os.path as osp

from bao.config import system_config, class_config


def split_df(df, train_names_file=osp.join(system_config.data_dir, "processed", "train_names.txt"), colname="fname"):
    np.random.seed(24)
    train_names = np.loadtxt(train_names_file, dtype='object')
    df_train = df[np.isin(df["fname"], train_names)]
    df_val = df[~np.isin(df["fname"], train_names)]
    return df_train, df_val

def filter_bad_mask_pred(df):
    return df[~df['id'].isin(class_config.bad_id_masks)]
