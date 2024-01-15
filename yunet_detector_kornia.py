# Fast face detector Yunet, benchmark p.7 https://link.springer.com/content/pdf/10.1007/s11633-023-1423-y.pdf

# See usage at # Sface: the fastest (also powerful) deep learning face recognition model in the world
# https://trungtranthanh.medium.com/sface-the-fastest-also-powerful-deep-learning-face-recognition-model-in-the-world-8c56e7d489bc

# However, opencv does not provide batch inference.

# Kornia code
# FaceDetector uses YUNet in batches https://kornia.readthedocs.io/en/latest/applications/face_detection.html

# TODO: Datector seems to have a leak of some sort, it crashes after some time when working with batch > 1

import kornia as K
import torch
import torch.nn.functional as F
import cv2
from kornia.contrib import FaceDetector, FaceDetectorResult, FaceKeypoint
import numpy as np
from PIL import Image


def draw_keypoint(img: np.ndarray, det: FaceDetectorResult, kpt_type: FaceKeypoint) -> np.ndarray:

    kpt = det.get_keypoint(kpt_type).int().tolist()

    return cv2.circle(img, kpt, 2, (255, 0, 0), 2)


class YUNetBatchKornia:
    def __init__(self, downsample=None, vis_threshold = 0.95, top_k=5000, confidence_threshold=0.1, nms_threshold=0.3, keep_top_k=750):

        # the parameters are default
        # downsample to speed up detection is float coefficient, e.g. 0.25
        # it don't see any resize under the hood https://github.com/kornia/kornia/blob/main/kornia/contrib/face_detection.py
        self.face_detection = FaceDetector(top_k=top_k, confidence_threshold=confidence_threshold, nms_threshold=nms_threshold, keep_top_k=keep_top_k)

        self.vis_threshold = vis_threshold
        self.downsample = downsample

    def detect(self, imgs_rgb):

        if isinstance(imgs_rgb, list):
            imgs_rgb = np.stack(imgs_rgb, axis=0)

        if isinstance(imgs_rgb, np.ndarray):
            imgs_rgb = K.image_to_tensor(imgs_rgb, keepdim=False).float()

        upsample_coeff = 1

        if self.downsample:
            downsample = self.downsample
            if downsample > 0 and downsample < 1:
                resize = (int(imgs_rgb.shape[2]*downsample), int(imgs_rgb.shape[3]*downsample))
                upsample_coeff = 1/downsample
                imgs_rgb = F.interpolate(imgs_rgb, size=resize, mode='bilinear')

        with torch.no_grad():
            dets = self.face_detection(imgs_rgb)

        res = []
        for d in dets:
            res.append([FaceDetectorResult(o) for o in d])

        # return # box coordinates and 5 landmarks
        final_res = []
        for idx, r in enumerate(res):
            res = []
            for b in r:
                if b.score < self.vis_threshold:
                    continue

                keypoint = [(b.get_keypoint(FaceKeypoint.EYE_LEFT)*upsample_coeff).tolist(),
                            (b.get_keypoint(FaceKeypoint.EYE_RIGHT)*upsample_coeff).tolist(),
                            (b.get_keypoint(FaceKeypoint.NOSE)*upsample_coeff).tolist(),
                            (b.get_keypoint(FaceKeypoint.MOUTH_LEFT)*upsample_coeff).tolist(),
                            (b.get_keypoint(FaceKeypoint.MOUTH_RIGHT)*upsample_coeff).tolist(),
                            ]

                res.append({'bbox':[(b.top_left*upsample_coeff).int().tolist(), (b.bottom_right*upsample_coeff).int().tolist()],
                            'lmk':keypoint})
            final_res.append(res)
        return final_res


if __name__ == '__main__':

    image = np.asarray(Image.open("frame0.png"))
    batch = np.stack([image,image],0)

    detector = YUNetBatchKornia()


    res = detector.detect([image, image])

    print(res)
