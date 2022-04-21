import cv2
import numpy as np
import torch
from detectron2.utils.visualizer import Visualizer, GenericMask, ColorMode
from detectron2.structures import BitMasks, Boxes, BoxMode, Keypoints, PolygonMasks, RotatedBoxes
from detectron2.utils.colormap import random_color

ClsName = {0: "H", 1: "B"}
def _create_text_labels(classes, body_ids, class_names):
    classes = classes.cpu().numpy().tolist()
    if body_ids is not None:
        labels = [ClsName[c] + "{:.0f}".format(s) for (c, s) in zip(classes, body_ids)]
    return labels
    
class CustomVisualizer(Visualizer):

    def modified_draw_instance_predictions(self, hand_boxes, hand_masks, hand_body_ids, body_boxes, body_masks, body_body_ids):
        all_body_ids = torch.cat([hand_body_ids, body_body_ids], dim=0)
        unique_ids = np.unique(all_body_ids.cpu().numpy()).tolist()
        colors = {}
        for id in unique_ids:
            colors[id] = random_color(rgb=True, maximum=1)
        
        hand_assigned_colors = []
        hand_body_ids_list = hand_body_ids.cpu().numpy().tolist()
        for body_id in hand_body_ids_list:
            hand_assigned_colors.append(colors[int(body_id)])

        hand_labels = ['H_' + str(int(hand_body_ids_list[i])) for i in range(hand_boxes.shape[0])]
        self.overlay_instances(
            masks=hand_masks,
            boxes=hand_boxes,
            labels=hand_labels,
            assigned_colors=hand_assigned_colors,
            alpha=0.3,
        )

        body_assigned_colors = []
        body_body_ids_list = body_body_ids.cpu().numpy().tolist()
        for body_id in body_body_ids_list:
            body_assigned_colors.append(colors[int(body_id)])
        body_labels = ['B_' + str(int(body_body_ids_list[i])) for i in range(body_boxes.shape[0])]
        self.overlay_instances(
            masks=body_masks,
            boxes=body_boxes,
            labels=body_labels,
            assigned_colors=body_assigned_colors,
            alpha=0.3,
        )
        return self.output

    def draw_dataset_dict(self, dic):

        annos = dic.get("annotations", None)
        if annos:
            if "segmentation" in annos[0]:
                masks = [x["segmentation"] for x in annos]
            else:
                masks = None
            if "keypoints" in annos[0]:
                keypts = [x["keypoints"] for x in annos]
                keypts = np.array(keypts).reshape(len(annos), -1, 3)
            else:
                keypts = None

            boxes = [BoxMode.convert(x["bbox"], x["bbox_mode"], BoxMode.XYXY_ABS) for x in annos]

            labels = [x["category_id"] for x in annos]
            colors = None
            if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
                colors = [
                    self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in labels
                ]
            names = self.metadata.get("thing_classes", None)
            if names:
                labels = [names[i] for i in labels]
            labels = [
                "{}".format(i) + ("|crowd" if a.get("iscrowd", 0) else "")
                for i, a in zip(labels, annos)
            ]
            labels = [ClsName[x["category_id"]] + '_' + str(x["body_id"]) for x in annos]
            ########## Modified here for color ####################
            body_ids = np.array([x["body_id"] for x in annos])
            unique_ids = np.unique(body_ids).tolist()
            colors = {}
            for id in unique_ids:
                colors[id] = random_color(rgb=True, maximum=1)
            assigned_colors = []

            body_ids_list = body_ids.tolist()
            for body_id in body_ids_list:
                assigned_colors.append(colors[int(body_id)])
            ##############################
            self.overlay_instances(
                labels=labels, boxes=boxes, masks=masks, keypoints=keypts, assigned_colors=assigned_colors, alpha=0.3,
            )

        sem_seg = dic.get("sem_seg", None)
        if sem_seg is None and "sem_seg_file_name" in dic:
            sem_seg = cv2.imread(dic["sem_seg_file_name"], cv2.IMREAD_GRAYSCALE)
        if sem_seg is not None:
            self.draw_sem_seg(sem_seg, area_threshold=0, alpha=0.5)
        return self.output