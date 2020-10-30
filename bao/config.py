import os.path as osp


CURRENT_PATH = osp.dirname(osp.realpath(__file__))

class SystemConfig():
    root_dir = osp.realpath(osp.join(CURRENT_PATH, ".."))
    model_dir = osp.join(root_dir, "models")
    data_dir = osp.join(root_dir, "data")

system_config = SystemConfig()
