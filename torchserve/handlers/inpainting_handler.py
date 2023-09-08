# handler.py
import sys
import os
import json
import torch
import logging
import numpy as np
from PIL import Image
import cv2
from ts.torch_handler.base_handler import BaseHandler

import zipfile
zipfile.ZipFile('./2401_80e_fine150e.zip').extractall()
zipfile.ZipFile('./src.zip').extractall()

from src.utils import numpy_to_b64, b64_to_numpy, resize_and_pad, recover_size
from diffusers import StableDiffusionInpaintPipeline
                
logger = logging.getLogger(__name__)


class InpaintingHandler(BaseHandler):
    def __init__(self):
        super(InpaintingHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        properties = ctx.system_properties        
        self.device = torch.device(f"cuda:{properties['gpu_id']}" if torch.cuda.is_available() else "cpu")
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained('./2401_80e_fine150e/').to(self.device)
        self.pipe.enable_attention_slicing()

        logger.debug("Model from path loaded successfully")
        self.initialized = True
        self.default_prompt = "photo of person, "

    def handle(self, data, ctx):       
        with torch.no_grad():
            # Load data from dictionary 
            dict_data = data[0]
            b64_image = dict_data['image']  
            b64_mask = dict_data['mask']
            text_prompt = self.default_prompt + dict_data['prompt']
            
            # Decode b64 images
            input_image = b64_to_numpy(b64_image)
            height, width, _ = input_image.shape
            mask = b64_to_numpy(b64_mask)
            
            # Resize image & mask
            image_padded, padding_factors = resize_and_pad(input_image)
            mask_padded, _ = resize_and_pad(mask, ismask=True)
            
            # Inpaint image
            image_inpainted = self.pipe(prompt=text_prompt, 
                                        image=Image.fromarray(image_padded), 
                                        mask_image=Image.fromarray(255-mask_padded),
                                        num_inference_steps=75,
                                        negative_prompt="duplicated features, watermark, disfigured, blurry, low quality"
                                        ).images[0]
            
            # Resize the image to the original size
            mask_padded = np.expand_dims(mask_padded[:,:,0], -1) / 255
            image_inpainted = image_inpainted * (1-mask_padded) + image_padded * mask_padded
            image_inpainted, mask_resized = recover_size(np.array(image_inpainted), mask_padded, (height, width), padding_factors)
            
            #mask = mask / 255
            #image_inpainted = image_inpainted * (1-mask) + input_image * mask
            image_inpainted = image_inpainted.astype(np.uint8)
            image_inpainted = cv2.cvtColor(image_inpainted, cv2.COLOR_RGB2BGR)
            b64_inpainted = numpy_to_b64(image_inpainted)
            
            message = "SUCCESS"
            result_dict = {'b64_inpainted': b64_inpainted,
                            'message': message}
            torch.cuda.empty_cache()
            return [json.dumps(result_dict)]
