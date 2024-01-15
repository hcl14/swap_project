
# Basic face swapping batch pipeline
# Face swapper prepares averaged embedding on init
# During inference, it accepts batch of images and returns batch of images

# I took ONNX models from facefusion and it turned out that they have batch dimension fixed = 1
# I adjusted ONNX file, but now onnx runtime complains that it's not optimal
# TODO: find initial .pth files and prepare good optimized ONNX files

import os
import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
import onnx
import onnxruntime as ort
from downloader import get_model
from util import load_images_from_folder, create_static_box_mask
from swap_models import templates, MODELS, face_embed_models, init_bisenet, load_gfpgan

from yunet_detector_kornia import YUNetBatchKornia
from warping_pipeline_cv import BatchWarper, warp_face

from video_util import VideoReader, VideoWriterFFmpeg

import time
from tqdm import tqdm

def get_model_matrix():
    model_path = model['path']
    model = onnx.load(model_path)
    MODEL_MATRIX = numpy_helper.to_array(model.graph.initializer[-1])
    return MODEL_MATRIX

# in facefusion, they calculate embedding based on fixed preset 'arcface_112_v2' https://github.com/facefusion/facefusion/blob/3e93f99eeb3f438dc416b1d82d91db742e791442/facefusion/face_analyser.py#L204

# I suppose, they think that pretrained arcface used this preset always.
# It's interesting question, because, as far as I remember training, they used the same warping for both target and source images
# so even if pretrained arcface has alignment 'arcface_112_v2', the training procedure supplied frames with different alignment

# but in this case the size will be 256, not 112 and arcface will fail. TODO: Need to examine training pipelines of those models
def calc_embedding(temp_frame, kps, face_recognizer, template):
    #crop_frame, matrix = warp_face(temp_frame, kps, 'arcface_112_v2', (112, 112))
    #crop_frame, matrix = bw.warp_face_single([temp_frame, kps]) # error: size 256 instead of 112
    crop_frame, matrix = warp_face(temp_frame, kps, template, 112)
    crop_frame = crop_frame.astype(np.float32) / 127.5 - 1
    crop_frame = crop_frame[:, :, ::-1].transpose(2, 0, 1)
    crop_frame = np.expand_dims(crop_frame, axis = 0)
    embedding = face_recognizer.run(None,
        {
            face_recognizer.get_inputs()[0].name: crop_frame
        })[0]
    embedding = embedding.ravel()
    normed_embedding = embedding / np.linalg.norm(embedding)
    return embedding, normed_embedding

def prepare_source_embedding(source_face_embedding, source_face_normed_embedding, model):
    if model['type'] == 'inswapper':
        model_matrix = get_model_matrix()
        source_embedding = source_face_embedding.reshape((1, -1))
        source_embedding = np.dot(source_embedding, model_matrix) / np.linalg.norm(source_embedding)
    else:
        source_embedding = source_face_normed_embedding.reshape(1, -1)
    return source_embedding


# class which contains face embedding, swap and refinement models
class FaceSwapper:
    def __init__(self, model:str, max_batch_size=5, source_folder = './source_frames'):

        self.model_name = model
        self.model = MODELS[model]
        self.mean = torch.tensor(np.array(self.model['mean'])[...,None, None]).float().cuda()
        self.std = torch.tensor(np.array(self.model['standard_deviation'])[...,None, None]).float().cuda()

        self.detector = YUNetBatchKornia(downsample=0.5) # faster

        self.source_folder = source_folder
        # image size to wrap into
        self.model_size = self.model['size']
        self.size = self.model_size[1]
        self.batch_size = max_batch_size

        # execute a node using CUDAExecutionProvider if capable, otherwise execute using CPUExecutionProvider
        self.face_swap_model = ort.InferenceSession(get_model(self.model), providers = ['CUDAExecutionProvider'])
        self.face_embed_model = ort.InferenceSession(get_model(face_embed_models[self.model['face_embed_model']]), providers = ['CUDAExecutionProvider'])
        self.gfpgan = load_gfpgan()

        if self.model['type'] == 'inswapper':
            self.model_matrix = get_model_matrix(model)

        self.compute_source_face()
        print(f"Initialized swapper with {self.model_name} model and {self.n_sources} source images")

        self.swap_time = 0
        self.preprocess_time = 0
        self.postprocess_time = 0


    def compute_source_face(self):
        # average embeddings of source faces in the folder

        # see comment above
        #bw = BatchWarper(self.model_name)
        template = templates[self.model['template']]

        frames_source = load_images_from_folder(self.source_folder)
        assert len(frames_source) != 0

        # extract faces from them
        source_embeddings = []
        source_normed_embeddings = []
        self.n_sources = 0
        for frame in frames_source:
            faces = self.detector.detect([frame])
            faces = faces[0]
            if len(faces) != 1:
                continue # discard uncertain images
            # embed source faces
            embedding, normed_embedding = calc_embedding(frame, np.array(faces[0]['lmk']), self.face_embed_model, template)
            source_embeddings.append(embedding)
            source_normed_embeddings.append(normed_embedding)
            self.n_sources += 1
        assert len(source_embeddings) > 0

        # average embeddings
        source_embedding = np.mean(source_embeddings, axis=0)
        source_normed_embedding = np.mean(source_normed_embeddings, axis=0)

        source_embedding = prepare_source_embedding(source_embedding, source_normed_embedding, self.model)
        self.source_embedding = np.concatenate([source_embedding]*self.batch_size, axis=0).astype(np.float32)

    def swap_batch(self, batch_faces, rgb=True, is_tensor=False):

        # swap preprocessing -------------------------
        start = time.time()
        batch = torch.tensor(np.array(batch_faces)).cuda().float()
        if not rgb:
            batch = batch[:, :, ::-1]
        if not is_tensor:
            batch = (batch.permute(0,3,1,2)/255. - self.mean) / self.std
        else:
            batch = (batch - self.mean) / self.std
        self.preprocess_time += time.time()-start

        '''
        # swapping batch -------------------------
        start = time.time()
        frame_processor_inputs = {}
        frame_processor_inputs['source'] = self.source_embedding[:batch.shape[0]] # batch can be smaller
        batch = batch.cpu().numpy()
        print(batch.shape)
        frame_processor_inputs['target'] = batch

        swap_result = self.face_swap_model.run(None, frame_processor_inputs)[0]
        self.swap_time += time.time()-start
        '''
        # swapping 1-by-1, because of the onnx model -------------------------
        start = time.time()
        swap_results = []
        for b, e in zip(batch, self.source_embedding[:batch.shape[0]]):
            frame_processor_inputs = {}
            frame_processor_inputs['source'] = e[None,...] # batch can be smaller
            b = b.cpu().numpy()
            frame_processor_inputs['target'] = b[None,...]


            swap_result = self.face_swap_model.run(None, frame_processor_inputs)[0]
            swap_results.append(swap_result)
        swap_result = np.concatenate(swap_results, axis=0)
        self.swap_time += time.time()-start
        # -----------------------------------------

        # swap postprocessing -------------------------
        swap_result = np.clip(swap_result * 255.0, 0, 255).astype(np.uint8) # on cpu

        if not rgb:
            swap_result = np.ascontiguousarray(swap_result[:, :, ::-1])
        return swap_result

    def enhance_batch(self, batch_frames, batch_orig, masks):
        # enhance upscaled faces using GFPGAN, blending source and final frames using bisenet mask
        # (No time to implement)
        # BS = 1

        # Quick and inefficient application of gfpgan (BGR)
        cropped_faces, restored_faces, res = self.gfpgan.enhance(np.array(batch_frames[0][...,::-1], dtype=np.uint8), has_aligned=False,
                                                                only_center_face=False, paste_back=True)

        return np.ascontiguousarray(res[...,::-1])



# Generate masks for blending
# Facefusion has occlusion model, I'm not using it
class FaceMasker:
    def __init__(self, model:str, max_batch_size=5, use_parsing_model=True):

        # use parsing model: use bisenet for face parsing and more accurate blending


        self.model_name = model
        self.model = MODELS[model]

        self.model_size = self.model['size']
        self.size = self.model_size[1]
        self.batch_size = max_batch_size

        self.use_parsing_model = use_parsing_model

        if use_parsing_model:
            # binent is quite robust to alignment (but it's better to realign)
            self.bisenet = init_bisenet(model_path='./models/parsing_bisenet.pth')

        # simple static mask
        static_mask = create_static_box_mask(crop_size=(self.model_size[1],self.model_size[1]))
        self.static_mask = np.stack([static_mask[None, ...]]*self.batch_size, axis=0)

    def preprocess_bisenet(self, batch_faces0):

        # currently ONLY FOR BATCH 1

        # we take mouth from original frame
        # and blend back after GFPGAN using mask, calculated on original frame

        batch_faces = torch.tensor(np.array(batch_faces0)).cuda().float().permute(0,3,1,2)
        batch_faces = F.interpolate(batch_faces, (512, 512))

        batch_faces = batch_faces / 255.
        face_input = normalize(batch_faces, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        with torch.no_grad():
            out = self.bisenet(face_input)[0]
        out = out.argmax(dim=1).squeeze().cpu().numpy() # 512x512


        masks_mouth = np.zeros(out.shape)
        MOUTH_COLORMAP = np.zeros(19)
        MOUTH_COLORMAP[12] = 255

        for idx, color in enumerate(MOUTH_COLORMAP):
            masks_mouth[out == idx] = color

        masks = np.zeros(out.shape)
        MASK_COLORMAP = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 0, 0, 0]

        for idx, color in enumerate(MASK_COLORMAP):
            masks[out == idx] = color

        """
        [(-1, 'unlabeled'), (0, 'background'), (1, 'skin'),
                 (2, 'l_brow'), (3, 'r_brow'), (4, 'l_eye'), (5, 'r_eye'),
                 (6, 'eye_g (eye glasses)'), (7, 'l_ear'), (8, 'r_ear'), (9, 'ear_r (ear ring)'),
                 (10, 'nose'), (11, 'mouth'), (12, 'u_lip'), (13, 'l_lip'),
                 (14, 'neck'), (15, 'neck_l (necklace)'), (16, 'cloth'),
                 (17, 'hair'), (18, 'hat')])
        """


        #cv2.imwrite('face0.png', batch_faces[0].permute(1,2,0).cpu().numpy()*255)
        #cv2.imwrite('mask0.png', masks_mouth)
        #cv2.imwrite('mask1.png', masks)

        return masks, masks_mouth


    def generate_masks(self, batch_images, indices):

        self.masks, self.masks_mouth = self.preprocess_bisenet(batch_images)

        static_masks = self.static_mask[indices]

        if not self.use_parsing_model:
            # just return simple masks
            masks = static_masks
            if len(masks.shape) == 3:
                masks = masks[None,...]
            return masks

        # combine static mask with face mask
        if len(static_masks.shape) == 3:
            staic_masks = static_masks[None,...]
        self.masks = cv2.resize(self.masks, (self.size,self.size))
        self.masks_mouth = cv2.resize(self.masks_mouth, (self.size,self.size))
        self.masks = (self.masks[None, None,...]/255.)*static_masks

        return self.masks, self.masks_mouth


# main swapping pipeline
# !!!! we limit to the first face per frame to simplify processing code
class MainPipeline:
    def __init__(self, model:str, max_batch_size=8):

        self.model_name = model
        self.model = MODELS[model]

        self.model_size = self.model['size']
        self.size = self.model_size[1]
        self.batch_size = max_batch_size

        self.batch_warper = BatchWarper(self.model_name)

        self.swapper = FaceSwapper(self.model_name, self.batch_size)
        self.masker = FaceMasker(self.model_name, self.batch_size)

        self.detector = self.swapper.detector #YUNetBatchKornia(downsample=0.25) # faster

        self.detector_time = 0
        self.waring_time = 0
        self.mask_generation_time = 0
        self.swap_time = 0
        self.swap_preprocess_time = 0
        self.swap_postprocess_time = 0
        self.blending_time = 0
        print("All initialized")

    def batch_process(self, batch):

        # detector -------------------------------
        # here we track frame indices where we found and processed faces
        good_frame_indices = []

        start = time.time()
        try:
            detected_faces = self.detector.detect(batch)
        except:
            print("Detector failed on batch, refrashing detector")
            detected_faces = []


        detected_faces_filtered = []
        for idx, r in enumerate(detected_faces):
            if len(r) != 0: # if there are faces, we take first one and mark this frame as good one
                good_frame_indices.append(idx)
                detected_faces_filtered.append(r[0])

        self.detector_time += time.time() - start
        # return empty result if no face found
        if len(detected_faces_filtered) == 0:
            return [], []

        batch_images = batch[good_frame_indices] # select subarray with frames with faces

        # warping ------------------------------
        start = time.time()
        #batch = batch.transpose(2,0,1)
        kps = [np.array(r['lmk']) for r in detected_faces_filtered]
        warped_frames, matrices = self.batch_warper.warp_batch(batch, kps)
        self.waring_time += time.time() - start

        # generating masks from source images -----------------------------
        start = time.time()
        blending_masks, masks_mouth = self.masker.generate_masks(warped_frames, good_frame_indices)

        self.mask_generation_time += time.time() - start

        # swapping --------------------------------------------------------
        swapped_faces = self.swapper.swap_batch(warped_frames)
        self.swap_time = self.swapper.swap_time # the time is computed on swapper side
        self.swap_preprocess_time = self.swapper.preprocess_time
        self.swap_postprocess_time = self.swapper.postprocess_time

        # blending swapped faces back ----------------------------
        start = time.time()
        '''
        print(batch_images.shape)
        print(swapped_faces.shape)
        print(blending_masks.shape)
        print(matrices)
        '''

        swapped_faces = swapped_faces.transpose(0,2,3,1)
        blending_masks = blending_masks.transpose(0,2,3,1)
        out_frames = self.batch_warper.warp_back_batch(batch_images, swapped_faces, blending_masks, matrices)
        self.blending_time += time.time() - start

        # if you want without GFPGAN
        #return out_frames


        # enhancing (GFPGAN) ----------------------------
        out_frames = self.swapper.enhance_batch(out_frames, batch_images, blending_masks)


        '''
        # with batch 1 it is not needed, need to debug
        # mix out_frames with unprocessed frames
        if len(good_frame_indices) < batch.shape[0]:
            batch[good_frame_indices] = out_frames
            return batch

        if isinstance(out_frames, np.ndarray):
            out_frames = out_frames.tolist()
        '''


        return [out_frames]

    def process(self, video_file, output_file):

        # reading video, can be in batches ---------------------------
        start = time.time()
        v = VideoReader(video_file)
        frames = v.frames[:]
        print("Video reading time", time.time() - start, "frames", len(frames))

        # processing video in batches --------------------------------
        out_frames = []
        for b_idx in tqdm(range(0, len(frames), self.batch_size)):
            batch = frames[b_idx*self.batch_size:(b_idx+1)*self.batch_size]
            out_frames += self.batch_process(batch)
        Image.fromarray(out_frames[0], mode='RGB').save('tmp.png')

        print("detector_time", self.detector_time)
        print("waring_time", self.waring_time)
        print("mask_generation_time", self.mask_generation_time)
        print("swap_time", self.swap_time)
        print("swap_preprocess_time", self.swap_preprocess_time)
        print("swap_postprocess_time", self.swap_postprocess_time)
        print("blending_time", self.blending_time)

        start = time.time()
        writer = VideoWriterFFmpeg(output_file, audio_source=video_file, resolution=(v.height, v.width), fps=v.fps, pix_fmt='rgb24') # 2 sec
        writer.imwrite(out_frames)
        writer.close()
        print("Writing video took", time.time() - start)











if __name__ == '__main__':

    pipe = MainPipeline('simswap_256', max_batch_size=1)
    pipe.process('../demo_input.mp4', 'demo_output.mp4')

'''
    fs = FaceSwapper('simswap_256')

    from yunet_detector_kornia import YUNetBatchKornia
    from PIL import Image

    image = np.asarray(Image.open("frame0.png"))

    detector = fs.detector

    batch = [image, image]
    res = detector.detect(batch)

    print(len(res))
    kps = [np.array(r[0]['lmk']) for r in res]

    bw = BatchWarper('simswap_256')

    res = bw.warp_batch(batch, kps)

    faces = res[0] #[r[0] for r in res]

    faces = np.stack(faces, axis=0).transpose(0,3,1,2)

    res = fs.swap_batch(faces)

    img = res[0].transpose(1,2,0)
    Image.fromarray(img).save('swapped.png')
'''

