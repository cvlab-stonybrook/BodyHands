import os
import xml.etree.ElementTree as ET
import numpy as np
from fvcore.common.file_io import PathManager
from typing import List, Tuple, Union
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

__all__ = ["load_voc_instances", "register_pascal_voc"]
CLASS_NAMES = ("hand", "body",)

def load_voc_instances(dirname: str, split: str, class_names: Union[List[str], Tuple[str, ...]]):
    
    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations/"))
    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(annotation_dirname, fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

        with PathManager.open(anno_file) as f:
            tree = ET.parse(f)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
 
        hand_annotations = {}
        body_annotations = {}
        for obj in tree.findall("object"):
            cls = obj.find("name").text
            bndbox = obj.find("bndbox")
            bbox = [float(bndbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            if cls == "hand":
                hand_px = [float(bndbox.find(x).text) for x in ["x1", "x2", "x3", "x4"]]
                hand_py = [float(bndbox.find(x).text) for x in ["y1", "y2", "y3", "y4"]]
                hand_poly = [(x, y) for x, y in zip(hand_px, hand_py)]
            else:
                body_px = [bbox[0], bbox[2], bbox[2], bbox[0]]
                body_py = [bbox[1], bbox[1], bbox[3], bbox[3]]
                body_poly = [(x, y) for x, y in zip(body_px, body_py)]
            body_id = int(obj.find("body_id").text)
            if cls == "hand":
                hand_ann = {
                        "category_id": class_names.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS,
                        "body_id": body_id, "segmentation": [hand_poly],
                    }
                if body_id in hand_annotations:
                    pass
                else:
                    hand_annotations[body_id] = []
                hand_annotations[body_id].append(hand_ann)
            else:
                body_ann = {
                     "category_id": class_names.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS,
                     "body_id": body_id, "segmentation": [body_poly], 
                    }
                if body_id in body_annotations:
                    pass 
                else:
                    body_annotations[body_id] = []
                body_annotations[body_id].append(body_ann)  
        
        instances = []
        for body_id in hand_annotations:
            body_ann = body_annotations[body_id][0]
            for hand_ann in hand_annotations[body_id]:
                hand_ann["body_box"] = body_ann["bbox"]
                instances.append(hand_ann)
            body_ann["body_box"] = body_ann["bbox"]
            instances.append(body_ann)

        r["annotations"] = instances
        dicts.append(r)

    return dicts

def register_pascal_voc(name, dirname, split, year, class_names=CLASS_NAMES):
    DatasetCatalog.register(name, lambda: load_voc_instances(dirname, split, class_names ))
    MetadataCatalog.get(name).set(
        thing_classes=CLASS_NAMES, dirname=dirname, year=year, split=split
    )

splits = ["train", "test"]
dirname = "./BodyHands/Data/VOC2007/"
for split in splits:
    register_pascal_voc("BodyHands_" + split , dirname, split, 2007)