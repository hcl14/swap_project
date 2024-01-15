# Fast face detector Yunet, benchmark p.7 https://link.springer.com/content/pdf/10.1007/s11633-023-1423-y.pdf

# See usage at # Sface: the fastest (also powerful) deep learning face recognition model in the world
# https://trungtranthanh.medium.com/sface-the-fastest-also-powerful-deep-learning-face-recognition-model-in-the-world-8c56e7d489bc

# However, opencv does not provide batch inference.



import onnxruntime as ort
import cv2
import numpy as np
import time
from itertools import product

import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.ops import nms

# output data format is described here https://github.com/geaxgx/depthai_yunet/blob/main/README.md
# postprocessing https://github.com/geaxgx/depthai_yunet/blob/main/models/build/generate_postproc_onnx.py
# pre-and postprocessing functions https://github.com/Kazuhito00/YuNet-ONNX-TFLite-Sample/blob/main/yunet/yunet_onnx.py#L30


# Feature map
iou_threshold = 0.3
score_threshold = 0.6
list_min_sizes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
steps = [8, 16, 32, 64]
variance = torch.from_numpy(np.array([0.1, 0.2]))
divisor = 32
top_k = 50

def prior_gen(w, h):
    feature_map_2th = [int(int((h + 1) / 2) / 2),
                        int(int((w + 1) / 2) / 2)]
    feature_map_3th = [int(feature_map_2th[0] / 2),
                        int(feature_map_2th[1] / 2)]
    feature_map_4th = [int(feature_map_3th[0] / 2),
                        int(feature_map_3th[1] / 2)]
    feature_map_5th = [int(feature_map_4th[0] / 2),
                        int(feature_map_4th[1] / 2)]
    feature_map_6th = [int(feature_map_5th[0] / 2),
                        int(feature_map_5th[1] / 2)]

    feature_maps = [feature_map_3th, feature_map_4th,
                    feature_map_5th, feature_map_6th]

    priors = []
    for k, f in enumerate(feature_maps):
        min_sizes = list_min_sizes[k]
        for i, j in product(range(f[0]), range(f[1])): # i->h, j->w
            for min_size in min_sizes:
                s_kx = min_size / w
                s_ky = min_size / h

                cx = (j + 0.5) * steps[k] / w
                cy = (i + 0.5) * steps[k] / h

                priors.append([cx, cy, s_kx, s_ky])

    priors = torch.from_numpy(np.array(priors, dtype=np.float32))
    return priors

class YunetPostProcessing(nn.Module):
    def __init__(self, w, h, top_k, priors):
        super(YunetPostProcessing, self).__init__()
        self.top_k = top_k
        self.priors = priors

    def forward(self, loc, conf, iou):
        # loc.shape: Nx14
        # conf.shape: Nx2
        # iou.shape: Nx1

        # get score
        cls_scores = conf[:,1]
        iou_scores = torch.squeeze(iou, 1)
        # clamp
        iou_scores[iou_scores < 0.] = 0.
        iou_scores[iou_scores > 1.] = 1.
        scores = torch.sqrt(cls_scores * iou_scores)
        # scores.unsqueeze_(1) # Nx1

        # get bboxes
        bb_cx_cy = self.priors[:, 0:2] + loc[:, 0:2] * variance[0] * self.priors[:, 2:4]
        bb_wh_half = self.priors[:, 2:4] * torch.exp(loc[:, 2:4] * variance) * 0.5
        bb_x1_y1 = bb_cx_cy - bb_wh_half
        bb_x2_y2 = bb_cx_cy + bb_wh_half
        bboxes = torch.cat((bb_x1_y1, bb_x2_y2), dim=1).float()

        # get landmarks
        landmarks = torch.cat((
            self.priors[:, 0:2] + loc[:,  4: 6] * variance[0] * self.priors[:, 2:4],
            self.priors[:, 0:2] + loc[:,  6: 8] * variance[0] * self.priors[:, 2:4],
            self.priors[:, 0:2] + loc[:,  8:10] * variance[0] * self.priors[:, 2:4],
            self.priors[:, 0:2] + loc[:, 10:12] * variance[0] * self.priors[:, 2:4],
            self.priors[:, 0:2] + loc[:, 12:14] * variance[0] * self.priors[:, 2:4]),
            dim=1)

        # NMS
        # TODO: torchvision.ops.batched_nms
        keep_idx = nms(bboxes, scores, iou_threshold)[:self.top_k]  # TODO: implement score threshold

        # scores = scores.unsqueeze(1)[keep_idx]
        dets = torch.cat((bboxes[keep_idx], landmarks[keep_idx], scores[keep_idx].unsqueeze_(1)), dim=1)
        return dets




class YUNetBatchMy:


    # opencv onnx model file available here (# there is int8 variant) https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet

    # But the models they used have fixed input shape either 160x120 or 640x640 which I found problematic making dynamic in python with onnx and onnxruntime
    # checked input shape with check_onnx script: 1, 3, 120, 160
    # when inference with arbitrary shape:
    # index: 2 Got: 1920 Expected: 640
    # index: 3 Got: 1088 Expected: 640

    # Python implementations I found just resize the inputs to 160x120 which will not allow detecting small faces

    # The older version face_detection_yunet_2021sep.onnx has dynamic inputs https://github.com/opencv/opencv_zoo/blob/5d155d8ec740a61a7a1964f5c3ecefe6a2b896a5/models/face_detection_yunet/face_detection_yunet_2021sep.onnx

    # There is prepocessing and postprocessing in c++ code: https://github.com/opencv/opencv/blob/b8e3bc9dd866b028e33b769e3c0992fc2b55a660/modules/objdetect/src/face_detect.cpp#L97-L121

    # input image is RGB



    def __init__(self, input_height, input_width): # input size is still needed to generate priors

        # make input shape divisable by 32
        padW = (int((input_width - 1) / divisor) + 1) * divisor;
        padH = (int((input_height - 1) / divisor) + 1) * divisor;

        self.input_shape = padW, padH

        # GPU needs CUDA 11.8, which is not in pip 3.11
        self.onnx_session = ort.InferenceSession('face_detection_yunet_2021sep.onnx', providers=['CUDAExecutionProvider']) #'CPUExecutionProvider',

        self.input_name = self.onnx_session.get_inputs()[0].name
        output_name_01 = self.onnx_session.get_outputs()[0].name
        output_name_02 = self.onnx_session.get_outputs()[1].name
        output_name_03 = self.onnx_session.get_outputs()[2].name
        self.output_names = [output_name_01, output_name_02, output_name_03]

        self.priors = prior_gen(padW, padH)
        self.postproc_model = YunetPostProcessing(padW, padH, top_k, self.priors)





    def infer(self, rgb_images):

        self.batch_size = rgb_images.shape[0] # save for postprocessing

        batch_cpu = self._preprocess_np_batch_gpu(rgb_images)

        result = self.onnx_session.run(
            self.output_names,
            {self.input_name: batch_cpu},
        )

        loc0, conf0, iou0 = result

        # we have coefficients from all images together. "Single Coefficient Size":
        scs = len(loc0)//self.batch_size

        N = self.priors.shape[0]

        dets_batch = []
        for b_idx in range(self.batch_size):
            loc = torch.tensor(loc0[b_idx*scs:(b_idx+1)*scs].reshape((N,14)))
            conf = torch.tensor(conf0[b_idx*scs:(b_idx+1)*scs].reshape((N,2)))
            iou = torch.tensor(iou0[b_idx*scs:(b_idx+1)*scs].reshape((N,1)))

            # remove low scoring boxes
            inds = torch.where(conf > score_threshold)[0]
            loc, conf, iou = loc[inds], conf[inds], iou[inds]

            result = self.postproc_model(loc, conf, iou)
            dets_batch.append(result)

        return dets_batch




    def _preprocess_np_batch_gpu(self, rgb_images, dtype=np.float32):

        # divisor is padding image dimensions to be multiple of
        # dtype is the input type needed by the model

        rgb_images = torch.tensor(rgb_images).cuda().float().permute((0,3,1,2))

        # this will not be needed if input image sizes are correct
        if rgb_images.shape[2] != self.input_shape[1] or rgb_images.shape[3] != self.input_shape[0]:

            start = time.time()
            rgb_images = F.interpolate(rgb_images, size=self.input_shape)
            print("Resizing images took", time.time()-start)

        images = rgb_images# No /255. # NCHW, float32 (can be uint8 for uint8 model)
        return images.cpu().numpy()



if __name__ == '__main__':

    img = cv2.imread('frame0.png') # BGR


    #x = preprocess(img)

    #outputs = ort_sess.run(None, {'input': x})

    img_batch = np.stack([img,img], axis=0)

    detector = YUNetBatchMy(img.shape[0], img.shape[1])

    bboxes = detector.infer(img_batch)

    print(bboxes)

    # https://github.com/geaxgx/depthai_yunet/blob/main/README.md
    # outputs consist of 3 arrays
    # takes the 3 outputs of the Yunet model (loc:Nx14, conf:Nx2 and iou:Nx1, with N depending on the Yunet input resolution, e.g. N=3210 for 180x360),
