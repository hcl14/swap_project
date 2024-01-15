
# Taken from https://github.com/facefusion/facefusion/blob/master/facefusion/face_analyser.py
import torch
import numpy
from bisenet import BiSeNet

from facexlib.utils import load_file_from_url
from gfpgan import GFPGANer

templates =\
{
    'arcface_112_v1': numpy.array(
    [
        [ 39.7300, 51.1380 ],
        [ 72.2700, 51.1380 ],
        [ 56.0000, 68.4930 ],
        [ 42.4630, 87.0100 ],
        [ 69.5370, 87.0100 ]
    ]),
    'arcface_112_v2': numpy.array(
    [
        [ 38.2946, 51.6963 ],
        [ 73.5318, 51.5014 ],
        [ 56.0252, 71.7366 ],
        [ 41.5493, 92.3655 ],
        [ 70.7299, 92.2041 ]
    ]),
    'arcface_128_v2': numpy.array(
    [
        [ 46.2946, 51.6963 ],
        [ 81.5318, 51.5014 ],
        [ 64.0252, 71.7366 ],
        [ 49.5493, 92.3655 ],
        [ 78.7299, 92.2041 ]
    ]),
    'ffhq_512': numpy.array(
    [
        [ 192.98138, 239.94708 ],
        [ 318.90277, 240.1936 ],
        [ 256.63416, 314.01935 ],
        [ 201.26117, 371.41043 ],
        [ 313.08905, 371.15118 ]
    ])
}

MODELS =\
{
    # Logic removed
    #'blendswap_256':
    #{
        #'type': 'blendswap',
        #'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/blendswap_256.onnx',
        #'path': './models/blendswap_256.onnx',
        #'template': 'ffhq_512',
        #'size': (512, 256),
        #'mean': [ 0.0, 0.0, 0.0 ],
        #'standard_deviation': [ 1.0, 1.0, 1.0 ],
        #'face_embed_model': 'face_recognizer_arcface_blendswap',
    #},
    'inswapper_128':
    {
        'type': 'inswapper',
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx',
        'path': './models/inswapper_128.onnx',
        'template': 'arcface_128_v2',
        'size': (128, 128),
        'mean': [ 0.0, 0.0, 0.0 ],
        'standard_deviation': [ 1.0, 1.0, 1.0 ],
        'face_embed_model': 'face_recognizer_arcface_inswapper',
    },
    'inswapper_128_fp16':
    {
        'type': 'inswapper',
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128_fp16.onnx',
        'path': './models/inswapper_128_fp16.onnx',
        'template': 'arcface_128_v2',
        'size': (128, 128),
        'mean': [ 0.0, 0.0, 0.0 ],
        'standard_deviation': [ 1.0, 1.0, 1.0 ],
        'face_embed_model': 'face_recognizer_arcface_inswapper',
    },
    'simswap_256':
    {
        'type': 'simswap',
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/simswap_256.onnx',
        'path': './models/simswap_256.onnx', # './models/simswap_256_N.onnx',
        'template': 'arcface_112_v1',
        'size': (112, 256),
        'mean': [ 0.485, 0.456, 0.406 ],
        'standard_deviation': [ 0.229, 0.224, 0.225 ],
        'face_embed_model': 'face_recognizer_arcface_simswap',
    },
}


face_embed_models = {
'face_recognizer_arcface_blendswap':
{
    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/arcface_w600k_r50.onnx',
    'path': './models/arcface_w600k_r50.onnx'
},
'face_recognizer_arcface_inswapper':
{
    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/arcface_w600k_r50.onnx',
    'path': './models/arcface_w600k_r50.onnx'
},
'face_recognizer_arcface_simswap':
{
    'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/arcface_simswap.onnx',
    'path': './models/arcface_simswap.onnx'
},

}

# taken from https://github.com/xinntao/facexlib/blob/master/facexlib/parsing/__init__.py#L8
def init_bisenet(half=False, device='cuda', model_path='./models/parsing_bisenet.pth'):
    model = BiSeNet(num_class=19)
    #model_url = 'https://github.com/xinntao/facexlib/releases/download/v0.2.0/parsing_bisenet.pth'

    #model_path = load_file_from_url(
    #    url=model_url, model_dir='facexlib/weights', progress=True, file_name=None, save_dir=model_rootpath)
    load_net = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(load_net, strict=True)
    model.eval()
    model = model.to(device)
    return model


# GFPGAN model paths here

# https://github.com/TencentARC/GFPGAN/blob/2eac2033893ca7f427f4035d80fe95b92649ac56/inference_gfpgan.py#L91

def load_gfpgan():
    gfpgan = GFPGANer(model_path='./models/GFPGANv1.4.pth', upscale=1, device='cuda')
    return gfpgan
