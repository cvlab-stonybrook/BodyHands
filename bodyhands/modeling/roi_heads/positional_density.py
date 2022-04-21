import torch
from torch import nn
from torch.nn import functional as F 
from detectron2.layers import Conv2d, Linear, ShapeSpec, cat, get_norm 
from detectron2.utils.registry import Registry
import numpy as np 
import fvcore.nn.weight_init as weight_init
from fvcore.nn import smooth_l1_loss
from detectron2.modeling.box_regression import Box2BoxTransform

ROI_POSITIONAL_DENSITY_HEAD_REGISTRY = Registry("ROI_POSITIONAL_DENSITY_HEAD")
ROI_POSITIONAL_DENSITY_HEAD_REGISTRY.__doc__ == ""

def PositionalDensityLoss(cfg, pred_mu_deltas, instances):

    proposal_boxes = []
    gt_corr_body_boxes = []
    gt_classes = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        gt_corr_body_boxes.append(instances_per_image.gt_body_boxes.tensor)
        proposal_boxes.append(instances_per_image.proposal_boxes.tensor)
        gt_classes.append(instances_per_image.gt_classes)
    if proposal_boxes:
        proposal_boxes = cat(proposal_boxes, dim=0)
        gt_corr_body_boxes = cat(gt_corr_body_boxes, dim=0)
        gt_classes = cat(gt_classes, dim=0)

    hand_proposal_boxes = proposal_boxes[gt_classes==0]
    gt_hand_corr_body_boxes = gt_corr_body_boxes[gt_classes==0]
    
    if hand_proposal_boxes.shape[0] == 0:
        return pred_mu_deltas.sum() * 0, pred_mu_deltas

    box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)
    gt_body_deltas = box2box_transform.get_deltas(
            hand_proposal_boxes, gt_hand_corr_body_boxes
        )
    smooth_l1_beta = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA # Using same default as box head
    loss = smooth_l1_loss(
            pred_mu_deltas,
            gt_body_deltas,
            smooth_l1_beta,
            reduction="mean",
        )
        
    weight = cfg.MODEL.ROI_POSITIONAL_DENSITY_HEAD.LOSS_WEIGHT
    loss = weight * loss

    pred_mu = box2box_transform.apply_deltas(
            pred_mu_deltas,
            hand_proposal_boxes,
        )

    return loss, pred_mu

def PositionalDensityInference(cfg, pred_mu_deltas, pred_instances):

    pred_boxes = []
    pred_classes = []
    for instances_per_image in pred_instances:
        if len(instances_per_image) == 0:
            continue
        pred_boxes.append(instances_per_image.pred_boxes.tensor)
        pred_classes.append(instances_per_image.pred_classes) # Assumes batchsize is 1
    if pred_boxes:
        pred_boxes = cat(pred_boxes, dim=0)
        pred_classes = cat(pred_classes, dim=0)
    else:
        pred_boxes = torch.empty(0, 4).to(cfg.MODEL.DEVICE)
        pred_classes = torch.empty(0).to(cfg.MODEL.DEVICE)
    pred_hand_boxes = pred_boxes[pred_classes==0]
    if not pred_hand_boxes.shape[0]:
        pred_hand_boxes = torch.empty(0, 4).to(cfg.MODEL.DEVICE)
    box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)
    pred_mu = box2box_transform.apply_deltas(
            pred_mu_deltas,
            pred_hand_boxes,
        )
    
    return pred_instances, pred_mu

@ROI_POSITIONAL_DENSITY_HEAD_REGISTRY.register()
class PositionalDensityHead(nn.Module):
    def __init__(self, cfg, input_shape: ShapeSpec):
        super(PositionalDensityHead, self).__init__()

        fc_dims = cfg.MODEL.ROI_POSITIONAL_DENSITY_HEAD.FC_DIM
        num_fc = len(fc_dims) 
        conv_params = cfg.MODEL.ROI_POSITIONAL_DENSITY_HEAD.CONV_DIMS
        conv_norm = cfg.MODEL.ROI_POSITIONAL_DENSITY_HEAD.CONV_NORM
        self.cfg = cfg 
        self.device = cfg.MODEL.DEVICE 
        self._output_size = (4+input_shape.channels, input_shape.height, input_shape.width)

        self.conv_norm_relus = []
        for k, conv_param in enumerate(conv_params):
            conv = Conv2d(
                self._output_size[0],
                conv_param[0],
                kernel_size=conv_param[1],
                padding=conv_param[2],
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_param[0]),
                activation=F.relu,
            )
            self.add_module("positional_density_conv{}".format(k+1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_param[0], self._output_size[1], self._output_size[2])

        self.fcs = []
        for k in range(num_fc):
            fc = Linear(np.prod(self._output_size), fc_dims[k])
            self.add_module("positional_density_fc{}".format(k+1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dims[k] 

        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer) 

    def forward(self, hand_boxes, hand_proposal_features, image_features, instances):
        
        hand_boxes = hand_boxes.unsqueeze(2).unsqueeze(2)
        hand_boxes = hand_boxes.repeat(1, 1, hand_proposal_features.shape[2], hand_proposal_features.shape[3])
        x = torch.cat([hand_proposal_features, hand_boxes], dim=1)
        for layer in self.conv_norm_relus:
            x = layer(x)
        
        x = torch.flatten(x, start_dim=1)
        for num in range(len(self.fcs)-1):
            x = F.relu(self.fcs[num](x))
        x = self.fcs[num+1](x)
        
        if self.training:
            loss, pred_mu = PositionalDensityLoss(self.cfg, x, instances)
            return {"positional density loss": loss}, pred_mu
        else:
            return PositionalDensityInference(self.cfg, x, instances)

def build_positional_density_head(cfg, input_shape):

    name = cfg.MODEL.ROI_POSITIONAL_DENSITY_HEAD.NAME 
    return ROI_POSITIONAL_DENSITY_HEAD_REGISTRY.get(name)(cfg, input_shape)    