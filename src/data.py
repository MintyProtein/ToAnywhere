import cv2
from PIL import Image, ImageDraw
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from utils import resize_and_pad


def random_mask(im_shape, ratio=1, mask_full_image=False):
    H, W, C = im_shape
    mask = Image.new("RGB", (W,H), (0,0,0))
    draw = ImageDraw.Draw(mask)
    size = (random.randint(0, int(W * ratio)), random.randint(0, int(H * ratio)))
    # use this to always mask the whole image
    if mask_full_image:
        size = (int(W * ratio), int(H * ratio))
    limits = (W - size[0] // 2, H - size[1] // 2)
    center = (random.randint(size[0] // 2, limits[0]), random.randint(size[1] // 2, limits[1]))
    draw_type = random.randint(0, 1)
    if draw_type == 0 or mask_full_image:
        draw.rectangle(
            (center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),
            fill=(1,1,1),
        )
    else:
        draw.ellipse(
            (center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),
            fill=(1,1,1),
        )
    mask = np.array(mask)
    return mask
        
class DefaultDataset(Dataset):
    def __init__(
        self,
        instance_image_paths,
        instance_mask_paths,
        instance_prompts,
        tokenizer,
        size=512,
    ):
        self.size = size
        self.tokenizer = tokenizer

        self.instance_image_paths = instance_image_paths
        self.instance_mask_paths = instance_mask_paths
        self.instance_prompts = instance_prompts
        
        if not len(self.instance_image_paths) == len(self.instance_mask_paths) == len(self.instance_prompts):
            print(f"# of images, masks, prompts are differenct. #images:{len(self.instance_image_paths)}, #masks:{len(self.instance_mask_paths)}, #prompts:{len(self.instance_prompts)}")
        self._length = len(self.instance_image_paths)
        
        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        instance_image = cv2.imread(self.instance_image_paths[index])
        instance_image = cv2.cvtColor(instance_image, cv2.COLOR_BGR2RGB)
        instance_image, _ = resize_and_pad(instance_image, target_size=self.size)
        instance_image = self.image_transforms(instance_image)
        instance_mask = cv2.imread(self.instance_mask_paths[index]).astype(np.float32) / 255
        if instance_mask.max() == 0:
            instance_mask = random_mask(instance_mask.shape)
        instance_mask[instance_mask>0.5] = 1
        instance_mask[instance_mask<=0.5] = 0
        instance_mask, _ = resize_and_pad(instance_mask, target_size=self.size, ismask=True)
        instance_mask = torch.from_numpy(instance_mask.transpose(2,0,1))
        instance_mask = 1 - instance_mask
        
        masked_image = instance_image * (instance_mask < 0.5)
        instance_mask = instance_mask[0,:,:][None, None,: :]
        instance_prompt_id = self.tokenizer(self.instance_prompts[index],
                                            padding='max_length',
                                            truncation=True,
                                            max_length=self.tokenizer.model_max_length,
                                            return_tensors='pt',
                                           ).input_ids[0]
        #instance_prompt_id = random_drop(instance_prompt_id)
        
        batch = {
            "pixel_values": instance_image,
            "masked_images": masked_image,
            "masks": instance_mask,
            "input_ids": instance_prompt_id
        }                           
        return batch
