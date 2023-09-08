# handler.py
import sys
import json
import zipfile
import torch
from torch.nn import functional as F
import logging
import numpy as np
import cv2
from ts.torch_handler.base_handler import BaseHandler

zipfile.ZipFile('./yolov7.zip').extractall()
zipfile.ZipFile('./src.zip').extractall()
zipfile.ZipFile('./mam.zip').extractall()
zipfile.ZipFile('./sam.zip').extractall()
sys.path.append('./yolov7')
sys.path.insert(0, './segment-anything')
from segment_anything.utils.transforms import ResizeLongestSide
from yolov7.models.experimental import attempt_load
from src.utils import numpy_to_b64, b64_to_numpy
from src.functions import predict_box_with_yolo, predict_mask_with_mam

import matting_anything as mam
from matting_anything.networks import get_generator_m2m

logger = logging.getLogger(__name__)


class YoloSamHandler(BaseHandler):
    def __init__(self):
        super(YoloSamHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        properties = ctx.system_properties        
        self.resolution=1024
        
        self.device = torch.device(f"cuda:{properties['gpu_id']}" if torch.cuda.is_available() else "cpu")
        self.transform = ResizeLongestSide(self.resolution)
        self.mam_model = get_generator_m2m(seg='sam_vit_l', m2m='sam_decoder_deep', sam_path='./sam_vit_l_0b8395.pth')
        self.mam_model.m2m.load_state_dict(mam.utils.remove_prefix_state_dict(torch.load('./mam_vitl.pth')['state_dict']), strict=True)
        self.mam_model.to(self.device)
        self.mam_model = self.mam_model.eval()

        self.yolo = attempt_load("./yolov7-e6e.pt", map_location=self.device)
        self.yolo.eval()
        self.initialized = True

    def handle(self, data, ctx):
        with torch.no_grad():  
            dict_data = data[0]
            message = "SUCCESS"
            print(type(dict_data))
            # Convert b64 string to Image
            b64_image = dict_data['image']
            input_img = b64_to_numpy(b64_image)
            
            original_size = input_img.shape[:2]
            image = self.transform.apply_image(input_img)
            image_tensor = torch.as_tensor(image)
            image_tensor = image_tensor.permute(2, 0, 1).contiguous()
            
            # Resize and pad the image
            h, w = image_tensor.shape[-2:]
            pad_size = image_tensor.shape[-2:]
            padh = self.resolution - h
            padw = self.resolution - w
            image_tensor = F.pad(image_tensor, (0, padw, 0, padh))
            
            # YOLO
            try:
                bbox = predict_box_with_yolo(self.yolo, image_tensor[None, :].to(self.device) / 255)
                bbox = torch.as_tensor(bbox, dtype=torch.float)[None, :]
            except:
                b64_mask = None
                message = "ERR_YOLO_UNKNOW"
            else:
                try:
                    # MAM
                    mask = predict_mask_with_mam(self.mam_model, image_tensor, bbox, original_size, pad_size, self.device)
                    b64_mask = numpy_to_b64(mask)
                except IndexError:
                    message = "ERR_HUMAN_NOT_DETECTED"
                    b64_mask = None
                
            result_dict = {'b64_mask': b64_mask,
                            'message': message}
            torch.cuda.empty_cache()
            return [json.dumps(result_dict)]
