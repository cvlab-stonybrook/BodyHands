import torch
from detectron2.layers import cat 
from bodyhands.utils.extend_utils_boxes import pairwise_ioa
from detectron2.structures import Boxes

def extract_handbody_components_training(features, roi_pooler, proposals):

    proposal_boxes = []
    gt_classes = []
    gt_body_ids = []
    gt_corr_body_boxes = []

    unique_gt_boxes = []
    unique_gt_classes = []
    unique_gt_body_ids = []
    unique_gt_boxes_roi_pooling = []  # These are of type Boxes, not Tensor
    for instances_per_image in proposals:
        if len(instances_per_image) == 0:
            continue
        proposal_boxes.append(instances_per_image.proposal_boxes)
        gt_corr_body_boxes.append(instances_per_image.gt_body_boxes.tensor)
        gt_boxes = instances_per_image.gt_boxes.tensor
        gt_classes.append(instances_per_image.gt_classes)
        gt_body_ids.append(instances_per_image.gt_body_ids)

        unique_boxes, unique_indices = torch.unique(gt_boxes, return_inverse=True, dim=0)
        reverse_unique_indices = []
        for o in unique_boxes:
            reverse_unique_indices.append((o == gt_boxes).all(1).nonzero()[0])
        reverse_unique_indices = torch.cat(reverse_unique_indices)

        unique_gt_boxes.append(unique_boxes)
        unique_gt_classes.append(instances_per_image.gt_classes[reverse_unique_indices])
        unique_gt_body_ids.append(instances_per_image.gt_body_ids[reverse_unique_indices])
        unique_gt_boxes_roi_pooling.append(instances_per_image.gt_boxes[reverse_unique_indices]) # These are of type Boxes, not Tensor

    gt_classes = cat(gt_classes, dim=0)
    gt_body_ids = cat(gt_body_ids, dim=0)

    unique_gt_classes = cat(unique_gt_classes, dim=0)
    unique_gt_boxes = cat(unique_gt_boxes, dim=0)
    unique_gt_hand_boxes = unique_gt_boxes[unique_gt_classes==0]
    unique_gt_body_boxes = unique_gt_boxes[unique_gt_classes==1]
    unique_gt_body_ids = cat(unique_gt_body_ids, dim=0)
    unique_gt_body_ids_hands = unique_gt_body_ids[unique_gt_classes==0]
    unique_gt_body_ids_bodies = unique_gt_body_ids[unique_gt_classes==1]
    
    unique_gt_features = roi_pooler(features, unique_gt_boxes_roi_pooling)
    unique_gt_hand_features = unique_gt_features[unique_gt_classes == 0]
    unique_gt_body_features = unique_gt_features[unique_gt_classes == 1]
    
    proposal_features = roi_pooler(features, proposal_boxes)
    hand_proposal_features = proposal_features[gt_classes==0]
    body_proposal_features = proposal_features[gt_classes==1]
    proposal_boxes = cat(proposal_boxes, dim=0).tensor
    hand_proposal_boxes = proposal_boxes[gt_classes==0]
    body_proposal_boxes = proposal_boxes[gt_classes==1]
    proposal_body_ids_hands = gt_body_ids[gt_classes==0]
    proposal_body_ids_bodies = gt_body_ids[gt_classes==1]
    gt_corr_body_boxes = cat(gt_corr_body_boxes, dim=0)
    gt_hand_corr_body_boxes = gt_corr_body_boxes[gt_classes==0]

    ioa_proposal_boxes = pairwise_ioa(Boxes(body_proposal_boxes), Boxes(hand_proposal_boxes)).T
    ioa_unique_gt_boxes = pairwise_ioa(Boxes(unique_gt_body_boxes), Boxes(unique_gt_hand_boxes)).T
    
    handbody_components = {
    "unique_gt_hand_boxes": unique_gt_hand_boxes, 
    "unique_gt_body_boxes": unique_gt_body_boxes,
    "unique_gt_body_ids_hands": unique_gt_body_ids_hands, 
    "unique_gt_body_ids_bodies": unique_gt_body_ids_bodies,
    "unique_gt_hand_features": unique_gt_hand_features,
    "unique_gt_body_features": unique_gt_body_features,
    "ioa_unique_gt_boxes": ioa_unique_gt_boxes,
    "hand_proposal_features": hand_proposal_features,
    "body_proposal_features": body_proposal_features,
    "hand_proposal_boxes": hand_proposal_boxes,
    "body_proposal_boxes": body_proposal_boxes,
    "gt_hand_corr_body_boxes": gt_hand_corr_body_boxes,
    "ioa_proposal_boxes": ioa_proposal_boxes,
    "proposal_body_ids_hands": proposal_body_ids_hands,
    "proposal_body_ids_bodies": proposal_body_ids_bodies,
    }

    return handbody_components

def extract_handbody_components_inference(pred_box_features, pred_boxes, pred_classes):

    hand_indices = pred_classes == 0
    body_indices = pred_classes == 1
    hand_boxes = pred_boxes[hand_indices]
    body_boxes = pred_boxes[body_indices]
    hand_features = pred_box_features[hand_indices]
    body_features = pred_box_features[body_indices]
    gt_ioa = pairwise_ioa(Boxes(body_boxes), Boxes(hand_boxes)).T
    handbody_components = {"hand_boxes": hand_boxes,
    "body_boxes": body_boxes,
    "hand_indices": hand_indices,
    "body_indices": body_indices,
    "hand_features": hand_features,
    "body_features": body_features,
    "gt_ioa": gt_ioa,
    }

    return handbody_components