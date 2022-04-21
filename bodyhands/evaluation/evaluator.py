import logging
import numpy as np
import os
import tempfile
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from functools import lru_cache
import torch
from fvcore.common.file_io import PathManager
from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.evaluation.evaluator import DatasetEvaluator

class CustomEvaluator(DatasetEvaluator):

    def __init__(self, dataset_name):

        self._dataset_name = dataset_name
        meta = MetadataCatalog.get(dataset_name)

        annotation_dir_local = PathManager.get_local_path(
            os.path.join(meta.dirname, "Annotations/")
        )
        self._anno_file_template = os.path.join(annotation_dir_local, "{}.xml")
        self._image_set_path = os.path.join(meta.dirname, "ImageSets", "Main", meta.split + ".txt")
        self._class_names = meta.thing_classes
        assert meta.year in [2007, 2012], meta.year
        self._is_2007 = meta.year == 2007
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

    def reset(self):
        self._predictions = defaultdict(list)  # class name -> list of prediction strings

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]
            instances = output["instances"].to(self._cpu_device)
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores
            classes = instances.pred_classes.numpy()
            body_ids = instances.pred_body_ids.numpy()

            hand_indices = (classes == 0)
            body_indices = (classes == 1)
            hand_scores = (scores[hand_indices]).tolist()
            hand_classes = classes[hand_indices]
            hand_boxes = boxes[hand_indices]
            if not hand_boxes.shape[0]:
                continue
            body_boxes = boxes[body_indices]
            hand_body_ids = body_ids[hand_indices]
            body_body_ids = body_ids[body_indices]
            num_hands = hand_boxes.shape[0]
            num_bodies = body_boxes.shape[0]
 
            hand_corr_body_boxes = []
            for hand_no in range(hand_boxes.shape[0]):
                id = hand_body_ids[hand_no]
                if id == -1 or id > num_bodies:
                    hand_corr_body_boxes.append(hand_boxes[hand_no])
                else:
                    hand_corr_body_boxes.append(body_boxes[body_body_ids==id])
            hand_corr_body_boxes = np.vstack(hand_corr_body_boxes)
            assert hand_boxes.shape == hand_corr_body_boxes.shape, 'hand has more than 1 body!'
            hand_classes = hand_classes.tolist()
            for box, score, cls, body_box in zip(hand_boxes, hand_scores, hand_classes, hand_corr_body_boxes):
                xmin, ymin, xmax, ymax = box
                xmin += 1
                ymin += 1
                body_xmin, body_ymin, body_xmax, body_ymax = body_box
                body_xmin += 1
                body_ymin += 1
                self._predictions[cls].append(
                    f"{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f} {body_xmin:.1f} {body_ymin:.1f} {body_xmax:.1f} {body_ymax:.1f}"
                )

    def evaluate(self):

        all_predictions = comm.gather(self._predictions, dst=0)
        if not comm.is_main_process():
            return
        predictions = defaultdict(list)
        for predictions_per_rank in all_predictions:
            for clsid, lines in predictions_per_rank.items():
                predictions[clsid].extend(lines)
        del all_predictions

        self._logger.info(
            "Evaluating {} using {} metric. "
            "Note that results do not use the official Matlab API.".format(
                self._dataset_name, 2007 if self._is_2007 else 2012
            )
        )

        with tempfile.TemporaryDirectory(prefix="pascal_voc_eval_") as dirname:
            res_file_template = os.path.join(dirname, "{}.txt")
            aps_dual_metric = defaultdict(list)
            aps_single_metric = defaultdict(list)
            for cls_id, cls_name in enumerate(self._class_names):
                if cls_name == "body":
                    continue
                lines = predictions.get(cls_id, [""])

                with open(res_file_template.format(cls_name), "w") as f:
                    f.write("\n".join(lines))

                for thresh in range(50, 51):
                    rec, prec, ap = voc_eval(
                        res_file_template,
                        self._anno_file_template,
                        self._image_set_path,
                        cls_name,
                        ovthresh=thresh / 100.0,
                        use_07_metric=self._is_2007,
                        single_metric=False,
                    )
                    aps_dual_metric[thresh].append(ap * 100)
                    rec, prec, ap = voc_eval(
                        res_file_template,
                        self._anno_file_template,
                        self._image_set_path,
                        cls_name,
                        ovthresh=thresh / 100.0,
                        use_07_metric=self._is_2007,
                        single_metric=True,
                    )
                    aps_single_metric[thresh].append(ap * 100)

        ret = OrderedDict()
        mAP_dual_metric = {iou: np.mean(x) for iou, x in aps_dual_metric.items()}
        mAP_single_metric = {iou: np.mean(x) for iou, x in aps_single_metric.items()}
        ret["bbox"] = {"AP@50IoU_dual_metric": mAP_dual_metric[50], "AP@50IoU_single_metric": mAP_single_metric[50]}
        return ret


##############################################################################
#
# Below code is modified from
# https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

"""Python implementation of the PASCAL VOC devkit's AP evaluation code."""


@lru_cache(maxsize=None)
def parse_rec(filename):
    """Parse a PASCAL VOC xml file."""
    with PathManager.open(filename) as f:
        tree = ET.parse(f)
    
    hand_annotations = {}
    body_annotations = {}
    for obj in tree.findall("object"):
        obj_struct = {}
        obj_struct["name"] = obj.find("name").text
        obj_struct["pose"] = obj.find("pose").text
        obj_struct["truncated"] = int(obj.find("truncated").text)
        obj_struct["difficult"] = int(obj.find("difficult").text)
        cls_ = obj.find("name").text
        bbox = obj.find("bndbox")
        obj_struct["bbox"] = [
            int(bbox.find("xmin").text),
            int(bbox.find("ymin").text),
            int(bbox.find("xmax").text),
            int(bbox.find("ymax").text),
        ]
        body_id = int(obj.find("body_id").text)
        if cls_ == "hand":
            if body_id in hand_annotations:
                    pass
            else:
                hand_annotations[body_id] = []
            hand_annotations[body_id].append(obj_struct)
        else:
            body_annotations[body_id] = [obj_struct] 

    objects = []
    for body_id in hand_annotations:
        body_ann = body_annotations[body_id][0]
        for hand_ann in hand_annotations[body_id]:
            hand_ann["body_box"] = body_ann["bbox"]
            objects.append(hand_ann)

    return objects


def voc_ap(rec, prec, use_07_metric=False):

    if use_07_metric:
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath, annopath, imagesetfile, classname, ovthresh=0.5, use_07_metric=False, single_metric=False):

    with PathManager.open(imagesetfile, "r") as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    recs = {}
    for imagename in imagenames:
        recs[imagename] = parse_rec(annopath.format(imagename))

    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj["name"] == classname]
        bbox = np.array([x["bbox"] for x in R])
        body_box = np.array([x["body_box"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {"bbox": bbox, "body_box": body_box, "difficult": difficult, "det": det}

    detfile = detpath.format(classname)
    with open(detfile, "r") as f:
        lines = f.readlines()

    splitlines = [x.strip().split(" ") for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:6]] for x in splitlines]).reshape(-1, 4)
    body_BB = np.array([[float(z) for z in x[6:]] for x in splitlines]).reshape(-1, 4)

    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    body_BB = body_BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    body_acc_count = 0
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        body_bb = body_BB[d, :].astype(float)
        ovmax = -np.inf
        body_ovmax = -np.inf
        BBGT = R["bbox"].astype(float)
        body_BBGT = R["body_box"].astype(float)

        if BBGT.size > 0:

            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:

                    body_bbgt_d = body_BBGT[jmax, :]

                    ixmin = np.maximum(body_bbgt_d[0], body_bb[0])
                    iymin = np.maximum(body_bbgt_d[1], body_bb[1])
                    ixmax = np.minimum(body_bbgt_d[2], body_bb[2])
                    iymax = np.minimum(body_bbgt_d[3], body_bb[3])
                    iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
                    ih = np.maximum(iymax - iymin + 1.0, 0.0)
                    inters = iw * ih

                    uni = (
                        (body_bb[2] - body_bb[0] + 1.0) * (body_bb[3] - body_bb[1] + 1.0)
                        + (body_bbgt_d[2] - body_bbgt_d[0] + 1.0) * (body_bbgt_d[3] - body_bbgt_d[1] + 1.0)
                        - inters
                        )

                    overlaps_body = inters / uni 
                    
                    if not single_metric:
                        tp[d] = 1.0
                        R["det"][jmax] = 1                    
                        if overlaps_body > 0.5:
                            body_acc_count += 1
                    else:
                        if overlaps_body > 0.5:
                            tp[d] = 1.0
                            R["det"][jmax] = 1 
                        else:
                            fp[d] = 1.0
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    if not single_metric:
        body_acc = (body_acc_count / max(tp)) * 100.0
        print("Body Accuracy corresponding to Dual Metric is:", round(body_acc, 4))
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    return rec, prec, ap