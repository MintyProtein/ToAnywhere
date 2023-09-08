import cv2
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from PIL import Image
import numpy as np

def resize_and_pad(image: np.ndarray, target_size: int = 512, ismask=False):
    height, width, _ = image.shape
    max_dim = max(height, width)
    scale = target_size / max_dim
    new_height = int(height * scale)
    new_width = int(width * scale)
    image_resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    pad_height = target_size - new_height
    pad_width = target_size - new_width
    top_pad = pad_height // 2
    bottom_pad = pad_height - top_pad
    left_pad = pad_width // 2
    right_pad = pad_width - left_pad
    constant_value = 1 if ismask else 0
    image_padded = np.pad(image_resized, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant', constant_values=constant_value)
    return image_padded, (top_pad, bottom_pad, left_pad, right_pad)

def recover_size(image_padded: np.ndarray, mask_padded: np.ndarray, orig_size: Tuple[int, int], 
                 padding_factors: Tuple[int, int, int, int]):
    h,w,c = image_padded.shape
    top_pad, bottom_pad, left_pad, right_pad = padding_factors
    image = image_padded[top_pad:h-bottom_pad, left_pad:w-right_pad, :]
    mask = mask_padded[top_pad:h-bottom_pad, left_pad:w-right_pad]
    image_resized = cv2.resize(image, orig_size[::-1], interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask, orig_size[::-1], interpolation=cv2.INTER_LINEAR)
    return image_resized, mask_resized

def b64_to_pil(b64_image: str) -> Image:
    img = base64.b64decode(b64_image)
    img = BytesIO(img)
    img = Image.open(img)
    return img

def b64_to_numpy(b64_image) -> np.array:
    img = base64.b64decode(b64_image)
    img = np.frombuffer(img, dtype=np.uint8)  
    img = cv2.imdecode(img, flags=cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def pil_to_b64(image: Image) -> str:
    image_file = BytesIO()
    image.save(image_file, format="JPEG")
    image_bytes = image_file.getvalue()  # im_bytes: image in binary format.
    return base64.b64encode(image_bytes).decode('utf8')

def numpy_to_b64(image: np.array) -> str:
    _, image_arr = cv2.imencode('.jpg', image)  # im_arr: image in Numpy one-dim array format.
    image_bytes = image_arr.tobytes()
    return base64.b64encode(image_bytes).decode('utf8')