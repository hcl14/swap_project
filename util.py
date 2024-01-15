import os
from PIL import Image
import numpy as np
import cv2

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        try:
            img = np.array(Image.open(os.path.join(folder,filename)).convert('RGB'))
            images.append(img)
        except:
            print("could not read", filename)
            continue
    return images

# simple static box mask from facefusion
def create_static_box_mask(crop_size=(256,256), face_mask_blur=0.3, face_mask_padding=[ 0, 0, 0, 0 ]):
    blur_amount = int(crop_size[0] * 0.5 * face_mask_blur)
    blur_area = max(blur_amount // 2, 1)
    box_mask = np.ones(crop_size, np.float32)
    box_mask[:max(blur_area, int(crop_size[1] * face_mask_padding[0] / 100)), :] = 0
    box_mask[-max(blur_area, int(crop_size[1] * face_mask_padding[2] / 100)):, :] = 0
    box_mask[:, :max(blur_area, int(crop_size[0] * face_mask_padding[3] / 100))] = 0
    box_mask[:, -max(blur_area, int(crop_size[0] * face_mask_padding[1] / 100)):] = 0
    if blur_amount > 0:
        box_mask = cv2.GaussianBlur(box_mask, (0, 0), blur_amount * 0.25)
    return box_mask
