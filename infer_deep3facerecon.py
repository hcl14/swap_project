import os
from PIL import Image
from tracker68.face_tracker import OfflineReader
from preprocess_deep3d import align_img
import cv2, numpy as np
from models_3dmm import create_model
from options.test_options import TestOptions
from scipy.io import loadmat
import torch

def load_lm3d():

    Lm3D = loadmat(os.path.join('BFM', 'similarity_Lm3D_all.mat'))
    Lm3D = Lm3D['lm']

    # calculate 5 facial landmarks using 68 landmarks
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    Lm3D = np.stack([Lm3D[lm_idx[0], :], np.mean(Lm3D[lm_idx[[1, 2]], :], 0), np.mean(
        Lm3D[lm_idx[[3, 4]], :], 0), Lm3D[lm_idx[5], :], Lm3D[lm_idx[6], :]], axis=0)
    Lm3D = Lm3D[[1, 2, 0, 3, 4], :]

    return Lm3D


def read_data(im_path, detector, lm3d_std, to_tensor=True):
    # to RGB
    im = Image.open(im_path).convert('RGB')
    W,H = im.size
    _,_,lm0 = detector.get_data(np.array(im),0) # 68 landmarks
    lm0= lm0.reshape([-1, 2])
    lm = lm0.copy()
    lm[:, -1] = H - 1 - lm[:, -1]

    print(im.size, lm.shape, lm3d_std.shape)
    _, im, lm, _ = align_img(im, lm, lm3d_std)
    if to_tensor:
        im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        lm = torch.tensor(lm).unsqueeze(0)
    return im, lm, lm0

opt = TestOptions().parse()
model = create_model(opt)
model.setup(opt)
model.device = 'cuda'
#model.parallelize()
model.eval()


lm3d_std = load_lm3d()
detector = OfflineReader()

im_tensor, lm_tensor, lm_orig = read_data('frame0.png', detector, lm3d_std)

# get warping transform which describes new landmarks of processed image
affine_matrix = cv2.estimateAffinePartial2D(lm_orig, lm_tensor.numpy()[0], method = cv2.RANSAC, ransacReprojThreshold = 100)[0]

data = {
            'imgs': im_tensor.cuda(),
            'lms': lm_tensor.cuda()
        }
model.set_input(data)

res_lm = model.forward()[0]

# get even better landmarks from 3dmm model

