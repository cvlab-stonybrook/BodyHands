from detectron2.data.build import build_detection_test_loader, build_detection_train_loader
from .dataset_mapper import CustomDatasetMapper

def custom_train_loader(cfg):
    return build_detection_train_loader(cfg, mapper=CustomDatasetMapper(cfg, True))
    
def custom_test_loader(cfg, dataset_name):
    return build_detection_test_loader(cfg, dataset_name, mapper=CustomDatasetMapper(cfg, False))