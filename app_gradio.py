import argparse
import sys
sys.path.append('./yolov7') 
sys.path.insert(0, './segment-anything')
import cv2
from PIL import Image
import numpy as np
import torch
from torch.nn import functional as F
import gradio as gr
from yolov7.models.experimental import attempt_load
sys.path.append('./segment-anything')
from segment_anything.utils.transforms import ResizeLongestSide
import matting_anything as mam
from matting_anything.networks import get_generator_m2m

from diffusers import StableDiffusionInpaintPipeline
from src import resize_and_pad, recover_size, save_image_mask
from src.functions import predict_box_with_yolo, predict_mask_with_mam


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Tobigs-19 Vision Conference")
    
    parser.add_argument('--yolo_path', default='./model_checkpoints/yolov7-e6e.pt')
    
    parser.add_argument('--mam_path', default='./model_checkpoints/mam_vitl.pth')
    
    parser.add_argument('--sam_path',  default='./model_checkpoints/sam_vit_l_0b8395.pth')
    
    parser.add_argument('--sd_path', default='./model_chehckpoints/2401_80e_fine150e/')
    
    parser.add_argument('--public_link', action='store_true', default=False)
    
    parser.add_argument('--resolution', default=1024) # Currently available on 1024 only
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else'cpu')
    
    with gr.Blocks() as demo:
        with torch.no_grad():
            transform = ResizeLongestSide(args.resolution)
            # yolo
            yolo = attempt_load(args.yolo_path, map_location=device)
            yolo.eval()
            
            # Matting anything
            mam_model = get_generator_m2m(seg='sam_vit_l', m2m='sam_decoder_deep', sam_path=args.sam_path)
            mam_model.m2m.load_state_dict(mam.utils.remove_prefix_state_dict(torch.load(args.mam_path)['state_dict']), strict=True)
            mam_model.to(device)
            mam_model = mam_model.eval()
            
            # Stable diffusion
            pipe = StableDiffusionInpaintPipeline.from_pretrained(args.sd_path, torch_dtype=torch.float32).to(device)
            
            with gr.Row():
                # Col A: Input image
                with gr.Column(scale=1):
                    gr.HTML("<h3><center>Input</center></h3>")
                    input_img = gr.Image(label='Input image', show_label=False).style(height=500)  
                            
                # Col B: Mask
                with gr.Column(scale=1):
                    gr.HTML("<h3><center>Input</center></h3>")
                    mask_img = gr.Image(label='Input image', show_label=False).style(height=500)
                    mask_btn = gr.Button(value='Get Mask')
                    
                #Col C: Inpainted image
                with gr.Column(scale=1):
                    gr.HTML("<h3><center>Output</center></h3>")
                    output_img = gr.Image(label='Inpainted image', interactive=False).style(height=500)
                    text_prompt = gr.Textbox(label='Text prompt')    
                    inpaint_btn = gr.Button(value='Inpaint')
            
            # reset components
            def reset_components():
                return None, None
            input_img.change(fn=reset_components,
                            outputs=[mask_img, output_img])
                            

            def on_mask_clicked(input_img):
                with torch.no_grad():
                    # Image preprocessing
                    original_size = input_img.shape[:2]
                    image = transform.apply_image(input_img)
                    image_tensor = torch.as_tensor(image)
                    image_tensor = image_tensor.permute(2, 0, 1).contiguous()
                    
                    # Resize and pad the image
                    h, w = image_tensor.shape[-2:]
                    pad_size = image_tensor.shape[-2:]
                    padh = args.resolution - h
                    padw = args.resolution - w
                    image_tensor = F.pad(image_tensor, (0, padw, 0, padh))
                    
                    # YOLO
                    bbox = predict_box_with_yolo(yolo, image_tensor[None, :].to(device) / 255)
                    bbox = torch.as_tensor(bbox, dtype=torch.float)[None, :]
        
                    # MAM
                    mask = predict_mask_with_mam(mam_model, image_tensor, bbox, original_size, pad_size, device)
                    
                    return mask
                
            mask_btn.click(fn=on_mask_clicked,
                           inputs=[input_img],
                           outputs=mask_img)
            
            def on_inpaint_clicked(input_img, mask, text_prompt):
                height, width, _ = input_img.shape
                
                # Resize image & mask
                image_padded, padding_factors = resize_and_pad(input_img)
                mask_padded, _ = resize_and_pad(mask, ismask=True)
                
                # Inpaint image
                image_inpainted = pipe(prompt=text_prompt, 
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
                return
            inpaint_btn.click(fn=on_inpaint_clicked,
                            inputs=[input_img, mask_img, text_prompt],
                            outputs=output_img)
    
        demo.launch(share=args.public_link, debug=False)