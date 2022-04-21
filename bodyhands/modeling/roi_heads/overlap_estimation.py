import torch
from torch import nn
from torch.nn import functional as F 
from detectron2.layers import Linear, ShapeSpec, Conv2d, get_norm, cat
from detectron2.utils.registry import Registry
import numpy as np 
import fvcore.nn.weight_init as weight_init
from detectron2.modeling.box_regression import Box2BoxTransform
from scipy.optimize import linear_sum_assignment

ROI_OVERLAP_ESTIMATION_HEAD_REGISTRY = Registry("ROI_OVERLAP_ESTIMATION_HEAD")
ROI_OVERLAP_ESTIMATION_HEAD_REGISTRY.__doc__ == """Registry for Overlap Estimation Module."""

def OverlapEstimationInference(cfg, handbody_components, pred_instances, device):
    
    num_hands = handbody_components["num_hands"]
    num_bodies = handbody_components["num_bodies"]
    hand_indices = handbody_components["hand_indices"]
    body_indices = handbody_components["body_indices"]
    gt_overlap = (handbody_components["gt_ioa"] > 0).float()
    
    if num_hands == 0:
        pred_instances[0].pred_body_ids = torch.Tensor([i for i in range(1, num_bodies+1)]).to(device)
        return pred_instances
    if num_bodies == 0:
        pred_instances[0].pred_body_ids = torch.Tensor([num_bodies+1] * num_hands).to(device)

    pred_body_ids = torch.Tensor([-1.0] * (num_hands+num_bodies)).to(device)
    pred_hand_boxes = handbody_components["hand_boxes"]
    pred_body_boxes = handbody_components["body_boxes"]
    pred_mu = handbody_components["pred_mu"]
    box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)
    mu_hand = box2box_transform.get_deltas(
            pred_hand_boxes, pred_mu
        )
    mu_body = [] # A list of length num_hands
    scores_positional_density = []
    for hand_no in range(num_hands):
        hand_boxes_hand_no = pred_hand_boxes[hand_no:hand_no+1]
        new_pred_body_boxes = torch.cat([pred_body_boxes, hand_boxes_hand_no], dim=0)
        hand_boxes_hand_no = hand_boxes_hand_no.repeat(num_bodies+1, 1)
        mu_body_hand_no = box2box_transform.get_deltas(
            hand_boxes_hand_no, new_pred_body_boxes
        ) # (num_bodies+1, 4)
        mu_hand_hand_no = mu_hand[hand_no:hand_no+1].repeat(num_bodies+1, 1) 
        # (Num_bodies+1, 4)
        conf_hand_no = torch.exp(
            -2.0 * 1e-1 * torch.sum(torch.abs(mu_hand_hand_no - mu_body_hand_no), dim=1)
        )
        scores_positional_density.append(conf_hand_no.reshape(1, num_bodies+1))
        mu_body.append(mu_body_hand_no)
    scores_positional_density = torch.cat(scores_positional_density, dim=0)
    pred_overlap = handbody_components["pred_overlap"]
    pred_overlap = F.sigmoid(pred_overlap)
    overlap_mask = (pred_overlap > 0.1).float()

    scores = pred_overlap * scores_positional_density * overlap_mask

    scores = torch.cat([scores, scores], dim=1)
    scores_numpy = scores.detach().to("cpu").numpy()
    row_ind, col_ind = linear_sum_assignment(-scores_numpy)
    col_ind = (col_ind % (num_bodies+1)) + 1
    row_ind, col_ind = torch.from_numpy(row_ind).to(device),\
    torch.from_numpy(col_ind).to(device)

    pred_body_ids_for_bodies = torch.arange(1, num_bodies+1).to(device)
    pred_body_ids_for_hands = torch.FloatTensor([num_bodies+1] * num_hands).to(device)
    pred_body_ids_for_hands[row_ind] = col_ind.float()
    pred_body_ids[hand_indices] = pred_body_ids_for_hands
    pred_body_ids[body_indices] = pred_body_ids_for_bodies.float()
    pred_instances[0].pred_body_ids = pred_body_ids

    return pred_instances

def OverlapEstimationLoss(pred_overlap, ioa_gt, cfg):
    weight = cfg.MODEL.ROI_OVERLAP_ESTIMATION_HEAD.LOSS_WEIGHT
    overlap_gt = (ioa_gt > 0).float()
    loss = weight * F.binary_cross_entropy_with_logits(pred_overlap, overlap_gt, reduction="mean")
    return loss


@ROI_OVERLAP_ESTIMATION_HEAD_REGISTRY.register()
class OverlapEstimationHead(nn.Module):

    def __init__(self, cfg, input_shape: ShapeSpec):
        
        super(OverlapEstimationHead, self).__init__()

        conv_params = cfg.MODEL.ROI_OVERLAP_ESTIMATION_HEAD.CONV_DIMS
        conv_norm = cfg.MODEL.ROI_OVERLAP_ESTIMATION_HEAD.CONV_NORM
        fc_dims = cfg.MODEL.ROI_OVERLAP_ESTIMATION_HEAD.FC_DIM
        num_fc = len(fc_dims)
        self.cfg = cfg 
        self.device = cfg.MODEL.DEVICE 
        self._output_size = (2*input_shape.channels, input_shape.height, input_shape.width)
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
            self.add_module("overlap_estimation_conv{}".format(k+1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_param[0], self._output_size[1], self._output_size[2])
        
        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)

        self.fcs = []
        for k in range(num_fc):
            fc = Linear(np.prod(self._output_size), fc_dims[k])
            self.add_module("overlap_estimation_fc{}".format(k+1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dims[k] 

        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer) 

    def forward(self, pred_mu, pred_mu_features, handbody_components, instances):

        if self.training:
            
            hand_proposal_features = handbody_components["hand_proposal_features"]
            body_proposal_features = handbody_components["body_proposal_features"]
            hand_proposal_boxes = handbody_components["hand_proposal_boxes"]
            body_proposal_boxes = handbody_components["body_proposal_boxes"]
            proposal_body_ids_hands = handbody_components["proposal_body_ids_hands"]
            proposal_body_ids_bodies = handbody_components["proposal_body_ids_bodies"]
            ioa_proposal_boxes = handbody_components["ioa_proposal_boxes"]
            num_hands = hand_proposal_boxes.shape[0]
            num_bodies = body_proposal_features.shape[0]

            if num_hands ==0 or num_bodies == 0:
                return {"loss overlap estimation": torch.sum(body_proposal_boxes) * 0,}

            pred_overlap = []
            for i in range(num_hands):
                h_f = hand_proposal_features[i: i+1]
                new_body_proposal_features = torch.cat([body_proposal_features, h_f], dim=0) 
                h_f = hand_proposal_features[i:i+1].repeat(num_bodies+1, 1, 1, 1)
                hb_f = torch.cat([h_f, new_body_proposal_features], dim=1)
                for num in range(len(self.conv_norm_relus)):
                    hb_f = self.conv_norm_relus[num](hb_f)
                hb_f = torch.flatten(hb_f, start_dim=1)
                for num in range(len(self.fcs)-1):
                    hb_f = F.relu(self.fcs[num](hb_f))
                if len(self.fcs) == 1:
                    num = -1
                hb_f = self.fcs[num+1](hb_f)
                hb_f = hb_f.squeeze(1).unsqueeze(0)
                pred_overlap.append(hb_f)
            pred_overlap = torch.cat(pred_overlap, dim=0)
            torch_ones = torch.ones(num_hands, 1).to(ioa_proposal_boxes.device)
            ioa_proposal_boxes = torch.cat([ioa_proposal_boxes, torch_ones], dim=1)
            return {"loss ioa prediction": OverlapEstimationLoss(pred_overlap, ioa_proposal_boxes, self.cfg),}
        
        else:
            pred_overlap = []
            hand_boxes = handbody_components["hand_boxes"]
            body_boxes = handbody_components["body_boxes"]
            hand_features = handbody_components["hand_features"]
            body_features = handbody_components["body_features"]
            num_hands = hand_boxes.shape[0]
            num_bodies = body_boxes.shape[0]
            for i in range(num_hands):
                h_f = hand_features[i: i+1]
                new_body_features = torch.cat([body_features, h_f], dim=0)
                h_f = hand_features[i:i+1].repeat(num_bodies+1, 1, 1, 1) 
                hb_f = torch.cat([h_f, new_body_features], dim=1)
                for num in range(len(self.conv_norm_relus)):
                    hb_f = self.conv_norm_relus[num](hb_f)
                hb_f = torch.flatten(hb_f, start_dim=1)
                for num in range(len(self.fcs)-1):
                    hb_f = F.relu(self.fcs[num](hb_f))
                if len(self.fcs) == 1:
                    num = -1
                hb_f = self.fcs[num+1](hb_f)
                hb_f = hb_f.squeeze(1).unsqueeze(0)
                pred_overlap.append(hb_f)
            if pred_overlap:
                pred_overlap = torch.cat(pred_overlap, dim=0)
                
            handbody_components["num_hands"] = num_hands
            handbody_components["num_bodies"] = num_bodies
            handbody_components["pred_overlap"] = pred_overlap
            handbody_components["pred_mu"] = pred_mu
            return OverlapEstimationInference(self.cfg, handbody_components, instances, self.device)

def build_overlap_estimation_head(cfg, input_shape):

    name = cfg.MODEL.ROI_OVERLAP_ESTIMATION_HEAD.NAME 
    return ROI_OVERLAP_ESTIMATION_HEAD_REGISTRY.get(name)(cfg, input_shape)        