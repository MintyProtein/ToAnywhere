import cv2
import numpy as np
import torch
from torch.nn import functional as F
from yolov7.utils.general import non_max_suppression 
from segment_anything.utils.transforms import ResizeLongestSide
from matting_anything.utils import get_unknown_box_from_mask, get_unknown_tensor_from_pred_oneside, get_unknown_tensor_from_mask_oneside


def predict_mask_with_sam(predictor, image, box):
    predictor.set_image(image)
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box,
        multimask_output=False,
    )
    predictor.reset_image()
    return masks[0].astype(np.uint8) * 255

def predict_mask_with_mam(mam_model, image_tensor, bbox, original_size, pad_size, device=torch.device("cuda")):  
    pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3,1,1)
    pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3,1,1)
    image_tensor = (image_tensor - pixel_mean) / pixel_std
    sample = {'image': image_tensor.unsqueeze(0).to(device), 
            'bbox': bbox.unsqueeze(0).to(device), 
            'ori_shape': original_size,
            'pad_shape': pad_size
            }

    feas, pred, post_mask = mam_model.forward_inference(sample)

    alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']
    alpha_pred_os8 = alpha_pred_os8[..., : sample['pad_shape'][0], : sample['pad_shape'][1]]
    alpha_pred_os4 = alpha_pred_os4[..., : sample['pad_shape'][0], : sample['pad_shape'][1]]
    alpha_pred_os1 = alpha_pred_os1[..., : sample['pad_shape'][0], : sample['pad_shape'][1]]

    alpha_pred_os8 = F.interpolate(alpha_pred_os8, sample['ori_shape'], mode="bilinear", align_corners=False)
    alpha_pred_os4 = F.interpolate(alpha_pred_os4, sample['ori_shape'], mode="bilinear", align_corners=False)
    alpha_pred_os1 = F.interpolate(alpha_pred_os1, sample['ori_shape'], mode="bilinear", align_corners=False)

    #weight_os8 = get_unknown_box_from_mask(post_mask)
    #alpha_pred_os8[weight_os8>0] = post_mask[weight_os8>0]
    #alpha_pred = alpha_pred_os8.clone().detach()

    weight_os8 = get_unknown_tensor_from_mask_oneside(post_mask, rand_width=10, train_mode=False)
    post_mask[weight_os8>0] = alpha_pred_os8[weight_os8>0]
    alpha_pred = post_mask.clone().detach()

    weight_os4 = get_unknown_tensor_from_pred_oneside(alpha_pred, rand_width=20, train_mode=False)
    alpha_pred[weight_os4>0] = alpha_pred_os4[weight_os4>0]

    weight_os1 = get_unknown_tensor_from_pred_oneside(alpha_pred, rand_width=10, train_mode=False)
    alpha_pred[weight_os1>0] = alpha_pred_os1[weight_os1>0]

    alpha_pred = alpha_pred[0][0].cpu().unsqueeze(-1).numpy()
    mask = cv2.cvtColor(np.uint8(alpha_pred*255), cv2.COLOR_GRAY2RGB)
    
    return mask


def predict_box_with_yolo(yolo, image_tensor):
    preds=yolo(image_tensor)[0]
    preds=preds[...,:6]   
    preds = non_max_suppression(torch.Tensor(preds), 0.25, 0.45, classes=[0], agnostic=False)[0].detach().cpu().numpy()
    return preds[0,:4]

