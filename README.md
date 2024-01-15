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
7.  **Writing resulting video**

## Video reading

Most solutions use frame-by-frame readers like moviepy to read frames into the RAM first. Fast alternative could be specialized fast batch loaders, specifically designed for deep learning tasks. Possible solutions are

1. **decord** - fast movie loader working in batches. able to use GPU. According to [the benchmark](https://github.com/bml1g12/benchmarking_video_reading_python) is one of the fastest solutions.
2. **NVidia DALI** (Links to methods and examples: [1](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/operations/nvidia.dali.fn.readers.video.html), [2](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/operations/nvidia.dali.fn.warp_affine.html), [3](https://github.com/NVIDIA/DALI/blob/main/docs/examples/math/geometric_transforms.ipynb)) uses GPU-accelerated codec and has fastest (to my knowledge) implementation of warp_affine operation which is needed to extract, align and blend faces back.

## Batch face detection and extraction

Most implementations rely on face detectors which work with one image at time. The idea was to take some fast face detector like Yunet, [benchmark on p.7]( https://link.springer.com/content/pdf/10.1007/s11633-023-1423-y.pdf) and infer it in batches, as detector models are usually lightweight and do not occupy much GPU resources.

### Attempt 1
Yunet is implemented in OpenCV [1](https://gist.github.com/UnaNancyOwen/3f06d4a0d04f3a75cc62563aafbac332), unfortunately, the implementation does not allow batch inference.

At first, I extracted ONNX model from OpenCV repo [link](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet), as well, as studied the [C++ code](https://github.com/opencv/opencv/blob/b8e3bc9dd866b028e33b769e3c0992fc2b55a660/modules/objdetect/src/face_detect.cpp#L97-L121).

Experiments with inferencing model via `onnxruntime` package  showed, that the serialized onnx model has fixed batch size 1 with input shapes expected to be 640x640, and setting it to different values in Python is tricky.

Checking issues revealed that they introduced fixed input model because of bugs, and the older version [face_detection_yunet_2021sep.onnx](https://github.com/opencv/opencv_zoo/blob/5d155d8ec740a61a7a1964f5c3ecefe6a2b896a5/models/face_detection_yunet/face_detection_yunet_2021sep.onnx) has dynamic inputs.

I found the [code](https://github.com/onnx/onnx/issues/2182) which makes batch size dynamic and converted the model successfully, but despite having correct outputs, unfortunately could not make NMS thresholding and overall postprocessing work correctly in time even though I found good documentation :
[output data format](https://github.com/geaxgx/depthai_yunet/blob/main/README.md),
[postprocessing](https://github.com/geaxgx/depthai_yunet/blob/main/models/build/generate_postproc_onnx.py),
[pre-and postprocessing functions](https://github.com/Kazuhito00/YuNet-ONNX-TFLite-Sample/blob/main/yunet/yunet_onnx.py#L30).

The class with my attempts is located in the file **yunet_detector_python_(postprocessing not working).py**. I think the task is doable, but it will be hard to achieve good performance of postprocessing code, as it has a lot of small operations.

### Attempt 2

Kornia has [this model implemented](https://kornia.readthedocs.io/en/latest/applications/face_detection.html) and seems to support batch input. The resulting batch face detector was implemented in **yunet_detector_kornia.py**. However, testing revealed that even though at the beginning everything seems fine, during execution the detector starts to skip frames and stops detecting faces after some number of batches. The behavior is present with batch > 1, so I had no choice except falling back to batch = 1 for this solution.



## Face parsing using Bisenet

![Face parsing](https://raw.githubusercontent.com/zllrunning/face-parsing.PyTorch/master/6.jpg)

Face parsing using BiSenet is implemented. Face parsing model has 19 classes which correspond to specific face parts. I has FFHQ alignment, but can be used with different alignment at the beginning.

