from torch.nn import functional as F
from detectron2.layers import paste_masks_in_image
from detectron2.structures import Instances

def detector_postprocess(results, output_height, output_width, mask_threshold=0.5):

    scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])
    results = Instances((output_height, output_width), **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(results.image_size)

    results = results[output_boxes.nonempty()]

    if results.has("pred_body_boxes"):
        output_body_boxes = results.pred_body_boxes
        output_body_boxes.scale(scale_x, scale_y)
        output_body_boxes.clip(results.image_size)
        results.pred_body_boxes = output_body_boxes

    if results.has("pred_masks"):
        results.pred_masks = paste_masks_in_image(
            results.pred_masks[:, 0, :, :],  # N, 1, M, M
            results.pred_boxes,
            results.image_size,
            threshold=mask_threshold,
        )


    return results