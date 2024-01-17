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


def read_data(im_rgb, detector, lm3d_std, to_tensor=True):
    # to RGB
    #im = Image.open(im_path).convert('RGB')
    #
    im = Image.fromarray(im_rgb, mode='RGB')
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


def extract_5p(lm):
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    lm5p = np.stack([lm[lm_idx[0], :], np.mean(lm[lm_idx[[1, 2]], :], 0), np.mean(
        lm[lm_idx[[3, 4]], :], 0), lm[lm_idx[5], :], lm[lm_idx[6], :]], axis=0)
    lm5p = lm5p[[1, 2, 0, 3, 4], :]
    return lm5p


def infer_3dmm(img_rgb):
    im_tensor, lm_tensor, lm_orig = read_data(img_rgb, detector, lm3d_std)

    data = {
                'imgs': im_tensor.cuda(),
                'lms': lm_tensor.cuda()
            }
    model.set_input(data)

    res_lm = model.forward().cpu().detach().numpy()[0]


    res_lm = extract_5p(res_lm)
    # get warping transform which describes new landmarks of processed image
    #affine_matrix = cv2.estimateAffinePartial2D(lm_tensor.numpy()[0], lm_orig, method = cv2.RANSAC, ransacReprojThreshold = 100)[0]

    #print(lm_orig.shape, lm_tensor.numpy()[0].shape)
    affine_matrix = cv2.getPerspectiveTransform(lm_tensor.numpy()[0].astype(np.float32)[:4,:], lm_orig.astype(np.float32)[:4,:])

    # project new landmarks back using this transform
    #inverse_matrix = cv2.invertAffineTransform(affine_matrix)

    #affine_matrix3x3 = np.eye(3)
    #affine_matrix3x3[:2,:] = affine_matrix


    better_lm = cv2.perspectiveTransform(res_lm[None, :, :], affine_matrix)

    print(res_lm[None, :, :].shape)
    print("!!!", better_lm)

    return better_lm[0].astype(np.int32)

if __name__ == '__main__':

    im = np.array(Image.open('frame0.png').convert('RGB'))

    better_lm = infer_3dmm(im)
    '''
    im_tensor, lm_tensor, lm_orig = read_data(im, detector, lm3d_std)

    data = {
                'imgs': im_tensor.cuda(),
                'lms': lm_tensor.cuda()
            }
    model.set_input(data)

    res_lm = model.forward()[0]

    # get even better landmarks from 3dmm model
    '''
    for (sX, sY) in better_lm:
        cv2.circle(im, (sX, sY), 1, (0, 0, 255), -1)

    Image.fromarray(im, mode='RGB').save('lmk.png')


