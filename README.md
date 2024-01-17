# Face swapping solution

Initial task was to produce a video face swapping code, trying to achieve one or two priorities:

1).  **Realistic result**
2) . **Code efficiency**


# Initial idea

Result quality is achieved through the following pipeline, with steps are described below:

0. **Video reading**
1. **Face detection and extraction** producing extracted and aligned face images
2.  **Face parsing** using BiSeNet and saving mouth mask and face area mask
3.  **Face swapping** using SimSwap
4.  **Blending mouth area from original face image** to remove mouth artifacts like double teeth
5.  **Restoring blended swap via GFPGAN** (the only frame consistent face autoencoder model)
6.  **Blending back enhanced swap using face area mask**
7.  **Using more robust landmarks from DeepFaceRecon model**
8.  **Writing resulting video**

## Running code

You need to make onnxruntime-gpu see your cuda. Versions 15-16 need cuda 11.4, the one installed via pip with pytorch does not work. I installed `cudatookit=11.8` in my conda env. Giving you `environments.yml` file.

Also download weights in the model folder, other weights the code should download on its own.

The main pipeline is being run by `swap_models.py`, check `if __name__ == '__main__':`


## Video reading

Most solutions use frame-by-frame readers like moviepy to read frames into the RAM first. Fast alternative could be specialized fast batch loaders, specifically designed for deep learning tasks. Possible solutions are

1. **decord** - fast movie loader working in batches. able to use GPU. According to [the benchmark](https://github.com/bml1g12/benchmarking_video_reading_python) is one of the fastest solutions. See implementation of video reading and writing in [**video_util.py**](https://github.com/hcl14/swap_project/blob/main/video_util.py)
2. **NVidia DALI** (Links to methods and examples: [1](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/operations/nvidia.dali.fn.readers.video.html), [2](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/operations/nvidia.dali.fn.warp_affine.html), [3](https://github.com/NVIDIA/DALI/blob/main/docs/examples/math/geometric_transforms.ipynb)) uses GPU-accelerated codec and has fastest (to my knowledge) implementation of warp_affine operation which is needed to extract, align and blend faces back.

## Batch face detection and extraction

Most implementations rely on face detectors which work with one image at time. The idea was to take some fast face detector like Yunet, [benchmark on p.7]( https://link.springer.com/content/pdf/10.1007/s11633-023-1423-y.pdf) and infer it in batches, as detector models are usually lightweight and do not occupy much GPU resources.

This detector detects 5 standard landmarks which can be used to align faces for Insightface, SimSwap and GFPGAN. Aligned face (BGR):

![Aligned face](https://github.com/hcl14/swap_project/blob/main/visuals/face.png)



### Attempt 1
Yunet is implemented in OpenCV [1](https://gist.github.com/UnaNancyOwen/3f06d4a0d04f3a75cc62563aafbac332), unfortunately, the implementation does not allow batch inference.

At first, I extracted ONNX model from OpenCV repo [link](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet), as well, as studied the [C++ code](https://github.com/opencv/opencv/blob/b8e3bc9dd866b028e33b769e3c0992fc2b55a660/modules/objdetect/src/face_detect.cpp#L97-L121).

Experiments with inferencing model via `onnxruntime` package  showed, that the serialized onnx model has fixed batch size 1 with input shapes expected to be 640x640, and setting it to different values in Python is tricky.

Checking issues revealed that they introduced fixed input model because of bugs, and the older version [face_detection_yunet_2021sep.onnx](https://github.com/opencv/opencv_zoo/blob/5d155d8ec740a61a7a1964f5c3ecefe6a2b896a5/models/face_detection_yunet/face_detection_yunet_2021sep.onnx) has dynamic inputs.

I found the [code](https://github.com/onnx/onnx/issues/2182) which makes batch size dynamic and converted the model successfully, but despite having correct outputs, unfortunately could not make NMS thresholding and overall postprocessing work correctly in time even though I found good documentation :
[output data format](https://github.com/geaxgx/depthai_yunet/blob/main/README.md),
[postprocessing](https://github.com/geaxgx/depthai_yunet/blob/main/models/build/generate_postproc_onnx.py),
[pre-and postprocessing functions](https://github.com/Kazuhito00/YuNet-ONNX-TFLite-Sample/blob/main/yunet/yunet_onnx.py#L30).

Also, converted model gets warnings from onnxruntime and is probably inefficient, as computation time is too high.

The class with my attempts is located in the file [**yunet_detector_python_(postprocessing not working).py**](https://github.com/hcl14/swap_project/blob/main/yunet_detector_python_(postprocessing%20not%20working).py). I think the task is doable, but it will be hard to achieve good performance of postprocessing code, as it has a lot of small operations.

### Attempt 2

Kornia has [this model implemented](https://kornia.readthedocs.io/en/latest/applications/face_detection.html) and seems to support batch input. The resulting batch face detector was implemented in [**yunet_detector_kornia.py**](https://github.com/hcl14/swap_project/blob/main/yunet_detector_kornia.py). However, testing revealed that even though at the beginning everything seems fine, during execution the detector starts to skip frames and stops detecting faces after some number of batches. The behavior is present with batch > 1, so I had no choice except falling back to batch = 1 for this solution. Further debugging is needed, or chice of another model. Kornia has .pth file in its code, so in theory it is possible to debug and compile good onnx model.



## Face parsing using Bisenet

![Face parsing](https://raw.githubusercontent.com/zllrunning/face-parsing.PyTorch/master/6.jpg)

Face parsing using BiSenet is implemented. Face parsing model has 19 classes which correspond to specific face parts. I has FFHQ alignment, but can be used with different alignment at the beginning.

```
[(-1, 'unlabeled'), (0, 'background'), (1, 'skin'),
(2, 'l_brow'), (3, 'r_brow'), (4, 'l_eye'), (5, 'r_eye'),
(6, 'eye_g (eye glasses)'), (7, 'l_ear'), (8, 'r_ear'), (9, 'ear_r (ear ring)'),
(10, 'nose'), (11, 'mouth'), (12, 'u_lip'), (13, 'l_lip'),
(14, 'neck'), (15, 'neck_l (necklace)'), (16, 'cloth'),
(17, 'hair'), (18, 'hat')])
```

2 masks are produced from aligned face: mouth mask and face area mask:

```
masks_mouth = np.zeros(out.shape)
MOUTH_COLORMAP = np.zeros(19)
MOUTH_COLORMAP[12] = 255

for idx, color in enumerate(MOUTH_COLORMAP):
    masks_mouth[out == idx] = color

masks = np.zeros(out.shape)
MASK_COLORMAP = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 0, 0, 0]

for idx, color in enumerate(MASK_COLORMAP):
    masks[out == idx] = color
```
![Mouth mask](https://github.com/hcl14/swap_project/blob/main/visuals/mask0.png)

![Face area mask](https://github.com/hcl14/swap_project/blob/main/visuals/mask1.png)



## Face swapping

Parts of the face swapping code were intially inspired by [facefusion](https://github.com/facefusion/facefusion), [because it allows to load multiple different swapping models](https://github.com/facefusion/facefusion/blob/3e93f99eeb3f438dc416b1d82d91db742e791442/facefusion/processors/frame/modules/face_swapper.py#L30), [2](https://github.com/facefusion/facefusion/blob/3e93f99eeb3f438dc416b1d82d91db742e791442/facefusion/face_analyser.py#L27), [having diffrent landmark templates for warping](https://github.com/facefusion/facefusion/blob/3e93f99eeb3f438dc416b1d82d91db742e791442/facefusion/face_helper.py#L11).

It is also able to use [GFPGAN as face enhancer](https://github.com/facefusion/facefusion/blob/3e93f99eeb3f438dc416b1d82d91db742e791442/facefusion/processors/frame/modules/face_enhancer.py#L51), also it [uses face occlusion model and different face parser](https://github.com/facefusion/facefusion/blob/3e93f99eeb3f438dc416b1d82d91db742e791442/facefusion/face_masker.py#L21).


The code uses SimSwap256, with potential to use other models. Some code for them was transferred from `facefusion`.

You can take a look at swapping class in [**swapping_pipeline.py**](https://github.com/hcl14/swap_project/blob/3a0f6dee4c14c28b45481a7004450729fccc27ac/swapping_pipeline.py#L70).

The code computes [mean embedding](https://github.com/hcl14/swap_project/blob/3a0f6dee4c14c28b45481a7004450729fccc27ac/swapping_pipeline.py#L102) of all the source pictures in `./source_frames` (facefusion does it also).


Function [`swap_batch`](https://github.com/hcl14/swap_project/blob/3a0f6dee4c14c28b45481a7004450729fccc27ac/swapping_pipeline.py#L135C9-L135C19) swaps target faces using source embedding. It can work with batches, but it is temporarily replaced with single image inference in the loop because of problems with models.

[`enhance_batch` function](https://github.com/hcl14/swap_project/blob/3a0f6dee4c14c28b45481a7004450729fccc27ac/swapping_pipeline.py#L183) should do restoration with GFPGAN and blending with masks, but is not implemented due to the lack of time. The code there applies GFPGAN to entire frame (though it is somewhat good, as restoreformer + gfpgan enhance antire image, so oversharpened swapped face is less visible).

Swap model has low resolution and is imperfect (see boundary and color artifacts):

![Swapped face (no gfpgan)](https://github.com/hcl14/swap_project/blob/main/visuals/swapped.png)

Entire pipeline is done in [MainPipeline class](https://github.com/hcl14/swap_project/blob/3a0f6dee4c14c28b45481a7004450729fccc27ac/swapping_pipeline.py#L318)


## Blending mouth area from original face image

Face swap models work bad with teeth. Blending mouth area is important to reduce transparent teeth, "double teeth" and other artifacts of SimSwap (though new artifacts appear when GFPGAN is applied to the image where mouth is not accurately blended.)

Currently mouth region mask obtained, but blending not implemented.


## Restoring blended swap via GFPGAN

GFPGAN can be applied on 512x512 face images, so deeper engineering is needed to re-warp 256x256 SimSwap outputs with simswap alignment into 512x512 GFPGAN outputs with FFHQ alignment. Then careful re-calculation must be done to warp those bigger face images back into original frame. I did that for Alias a 1,5 years ago and my code used NVidia Dali. However, implementing it now seems out of time frame provided.

Also, application of some restoration model (e.g. RealESRGAN) over the entire frame increases its sharpness and makes increased sharpness of the face less visible. However, GFPGAN significantly reduces face detail.

Here is frame without and with GFPGAN:

![Swapped face (no gfpgan)](https://github.com/hcl14/swap_project/blob/main/visuals/tmp_no_gfpgan.png)
![Swapped face (with gfpgan)](https://github.com/hcl14/swap_project/blob/main/visuals/tmp_gfpgan.png)

Also, the better is landmark detection, the more stable are faces.



## Blending back enhanced swap using face area mask

![before and after masking mouth](https://github.com/hcl14/swap_project/blob/main/results/comparsion_gfpgan_before_after_mouth_mask.png)

Implemented. See `results/demo_output_mouth_mask.mp4` and `results/comparison.mp4` in [**/results**](https://github.com/hcl14/swap_project/blob/main/results).


## Writing resulting video

Done very fast via ffmpeg wrapper with custom settings, also audio is copied from the souce clip. [See VideoWriter code](https://github.com/hcl14/swap_project/blob/ed5d6552776d46c01acf35eb3374e5b2a1ddbe38/video_util.py#L93)


## Using more robust landmarks from DeepFaceRecon model

As you can see on `results/comparison.mp4`, landmarks are not robust and aligned face is shifting wildly. Thanks to the high regularization of GFPGAN, the results are still robust to keep frame consistency. Yet you can see face slightly shifting. To improve that, we can take better landmarks predicted by face mesh recostruction models - either by Mediapipe face mesh, or 3DMM models like [Deep3dFaceRecon](https://github.com/sicxu/Deep3DFaceRecon_pytorch).

Currently, I am implementing Deep3DFacerecon approach, will share the results.

Weights for Deep3Dfacerecon: [https://drive.google.com/drive/folders/1i0LxjlvzgZ9Avx1Y4-Imn3yf8JvJu6o7?usp=sharing](https://drive.google.com/drive/folders/1i0LxjlvzgZ9Avx1Y4-Imn3yf8JvJu6o7?usp=sharing). Please put BFM folder to the project root, and BFM\checkpoints folder to ./checkpoints.

Currently `infer_deep3facerecon.py` inferences Deep3DfaceRecon and obtains precise landmarks.

![Precise landmarks](https://github.com/hcl14/swap_project/blob/main/visuals/lmk.png)


# Download video resluts:

[**/results**](https://github.com/hcl14/swap_project/blob/main/results)

