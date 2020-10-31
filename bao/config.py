import os.path as osp


CURRENT_PATH = osp.dirname(osp.realpath(__file__))

class SystemConfig():
    root_dir = osp.realpath(osp.join(CURRENT_PATH, ".."))
    model_dir = osp.join(root_dir, "models")
    data_dir = osp.join(root_dir, "data")

class ClassConfig(object):
    bad_id_masks = ['00007882_001_2', '00000211_041_2', '00011355_011_2']

system_config = SystemConfig()
class_config = ClassConfig()