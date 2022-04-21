import torch
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.poolers import ROIPooler
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads import select_foreground_proposals
from detectron2.structures import Boxes
from .extract_handbody_components import extract_handbody_components_training, extract_handbody_components_inference
from .overlap_estimation import build_overlap_estimation_head
from .positional_density import build_positional_density_head

@ROI_HEADS_REGISTRY.register()
class HandBodyROIHeads(StandardROIHeads): 

    def __init__(self, cfg, input_shape):
        super(HandBodyROIHeads, self).__init__(cfg, input_shape)
        self._init_positional_density_head(cfg, input_shape)
        self._init_overlap_estimation_head(cfg, input_shape)
        self.config = cfg

    def _init_positional_density_head(self, cfg, input_shape):

        pooler_resolution = cfg.MODEL.ROI_POSITIONAL_DENSITY_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_POSITIONAL_DENSITY_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_POSITIONAL_DENSITY_HEAD.POOLER_TYPE
        self.positional_density_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        in_channels = [input_shape[f].channels for f in self.in_features][0]
        self.positional_density_head = build_positional_density_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution))

    def _init_overlap_estimation_head(self, cfg, input_shape):

        pooler_resolution = cfg.MODEL.ROI_OVERLAP_ESTIMATION_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_OVERLAP_ESTIMATION_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_OVERLAP_ESTIMATION_HEAD.POOLER_TYPE
        self.overlap_estimation_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        in_channels = [input_shape[f].channels for f in self.in_features][0]
        self.overlap_estimation_head = build_overlap_estimation_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution))
        

    def _forward_positional_density(self, height, width, features, instances):

        image_box = torch.FloatTensor([[0, 0, width, height]])
        image_box = Boxes(image_box).to(self.config.MODEL.DEVICE)
        features = [features[f] for f in self.in_features]
        image_features = self.positional_density_pooler(features, [image_box])
        if self.training:
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            handbody_components = extract_handbody_components_training(features, self.positional_density_pooler, proposals)
            hand_proposal_features = handbody_components["hand_proposal_features"]
            hand_proposal_boxes = handbody_components["hand_proposal_boxes"]
            return self.positional_density_head(hand_proposal_boxes, hand_proposal_features, image_features, proposals)
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            pred_box_features = self.positional_density_pooler(features, pred_boxes)
            pred_boxes = pred_boxes[0].tensor # Assumes batchsize 1
            pred_classes = [x.pred_classes for x in instances]
            pred_classes = pred_classes[0] # Assumes batchsize 1
            handbody_components = extract_handbody_components_inference(
                pred_box_features, pred_boxes, pred_classes
                )
            hand_features = handbody_components["hand_features"]
            hand_boxes = handbody_components["hand_boxes"]
            return self.positional_density_head(hand_boxes, hand_features, image_features, instances)

    def _forward_overlap_estimation(self, pred_mu, pred_mu_features, features, instances):

        features = [features[f] for f in self.in_features]
        if self.training:
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            handbody_components = extract_handbody_components_training(features, self.overlap_estimation_pooler, proposals)
            return self.overlap_estimation_head(pred_mu, pred_mu_features, handbody_components, proposals)
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            pred_box_features = self.overlap_estimation_pooler(features, pred_boxes)
            pred_boxes = pred_boxes[0].tensor # Assumes batchsize 1
            pred_classes = [x.pred_classes for x in instances]
            pred_classes = pred_classes[0] # Assumes batchsize 1
            handbody_components = extract_handbody_components_inference(
                pred_box_features, pred_boxes, pred_classes
                )
            return self.overlap_estimation_head(pred_mu, pred_mu_features, handbody_components, instances)

    def forward(self, images, height, width, features, proposals, targets=None):
        
        del images
        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            losses.update(self._forward_mask(features, proposals))
            positional_density_loss, pred_mu = self._forward_positional_density(height, width, features, proposals)
            losses.update(positional_density_loss)
            pred_mu_pooling = Boxes(pred_mu)
            pred_mu_pooling.clip((height, width))
            pred_mu_features = self.overlap_estimation_pooler([features[f] for f in self.in_features], [pred_mu_pooling])
            losses.update(self._forward_overlap_estimation(pred_mu, pred_mu_features, features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            pred_instances = self.forward_with_given_boxes(height, width, features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(self, height, width, features, instances):

        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")
        instances = self._forward_mask(features, instances)
        instances, pred_mu = self._forward_positional_density(height, width, features, instances)
        pred_mu_pooling = Boxes(pred_mu)
        pred_mu_pooling.clip((height, width))
        pred_mu_features = self.overlap_estimation_pooler([features[f] for f in self.in_features], [pred_mu_pooling])        
        instances = self._forward_overlap_estimation(pred_mu, pred_mu_features, features, instances)
        return instances