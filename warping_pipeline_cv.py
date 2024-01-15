
# OnenCV warping functions as baseline. To be replaced with optimal GPU ones

# warping fuctions taken from https://github.com/facefusion/facefusion/blob/master/facefusion/face_helper.py
# https://github.com/facefusion/facefusion/blob/master/facefusion/processors/frame/modules/face_swapper.py

# Then warping utility class written

from typing import Any, Dict, Tuple, List
from cv2.typing import Size
import cv2
import numpy as np
from swap_models import templates, MODELS
import multiprocessing as mp

def warp_face(temp_frame : np.ndarray, kps : np.ndarray, normed_template : np.ndarray, size : int):
    affine_matrix = cv2.estimateAffinePartial2D(kps, normed_template, method = cv2.RANSAC, ransacReprojThreshold = 100)[0]
    crop_frame = cv2.warpAffine(temp_frame, affine_matrix, (size, size), borderMode = cv2.BORDER_REPLICATE)
    return crop_frame, affine_matrix


def paste_back(temp_frame : np.ndarray, crop_frame: np.ndarray, crop_mask : np.ndarray, affine_matrix : np.ndarray):
    inverse_matrix = cv2.invertAffineTransform(affine_matrix)
    temp_frame_size = temp_frame.shape[:2][::-1]
    inverse_crop_mask = cv2.warpAffine(crop_mask, inverse_matrix, temp_frame_size).clip(0, 1)
    inverse_crop_frame = cv2.warpAffine(crop_frame, inverse_matrix, temp_frame_size, borderMode = cv2.BORDER_REPLICATE)
    paste_frame = temp_frame.copy()
    paste_frame[:, :, 0] = inverse_crop_mask * inverse_crop_frame[:, :, 0] + (1 - inverse_crop_mask) * temp_frame[:, :, 0]
    paste_frame[:, :, 1] = inverse_crop_mask * inverse_crop_frame[:, :, 1] + (1 - inverse_crop_mask) * temp_frame[:, :, 1]
    paste_frame[:, :, 2] = inverse_crop_mask * inverse_crop_frame[:, :, 2] + (1 - inverse_crop_mask) * temp_frame[:, :, 2]
    return paste_frame


# simple class which uses multiprocessing to extract faces, warping them based on the model settings
class BatchWarper:
    def __init__(self, model_used:str):
        model = MODELS[model_used]
        # image size to wrap into
        self.model_size = model['size']
        self.size = self.model_size[1]
        # kps used to wrap into
        keypoints_template = templates[model['template']]
        self.normed_template = keypoints_template * self.model_size[1] / self.model_size[0]


    def warp_face_single(self, data):
        img, kps = data
        return warp_face(img, kps, self.normed_template, self.size)

    def warp_batch(self, images, kps):
        payload = list(zip(images, kps))

        # use all available CPUs
        p = mp.Pool()
        res = list(p.imap(self.warp_face_single, payload))

        res_images = [r[0] for r in res]
        res_matrices = [r[1] for r in res]


        p.close()
        p.join()

        return [res_images, res_matrices]

    def warp_back_single(self, data):
        image, face, mask, matrix = data
        return paste_back(image, face, mask, matrix)

    def warp_back_batch(self, images, faces, masks, matrices):

        payload = list(zip(images, faces, masks, matrices))

        # use all available CPUs
        p = mp.Pool()
        res = list(p.imap(self.warp_back_single, payload))

        p.close()
        p.join()

        return res






if __name__ == '__main__':

    from yunet_detector_kornia import YUNetBatchKornia
    from PIL import Image

    image = np.asarray(Image.open("frame0.png"))

    detector = YUNetBatchKornia()

    batch = [image, image]
    res = detector.detect(batch)

    kps = [np.array(r[0]['lmk']) for r in res]

    bw = BatchWarper('simswap_256')

    res = bw.warp_batch(batch, kps)

    face = res[0][0]
    cv2.imwrite('face.png', face)

    mask = np.ones(face.shape[:2], dtype=np.uint8)

    faces = res[0] #[r[0] for r in res]
    images = batch
    masks = [mask, mask]
    matrices = res[1] #[r[1] for r in res]

    result = bw.warp_back_batch(images, faces, masks, matrices)

    cv2.imwrite('warp_back.png', result[0])




