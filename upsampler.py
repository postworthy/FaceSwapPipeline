import os
import cv2
os.environ['OMP_NUM_THREADS'] = '1'
import threading
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer #https://github.com/postworthy/GFPGAN


THREAD_LOCK_UPSAMPLER = threading.Lock()
THREAD_LOCK_UPSAMPLER_FAST = threading.Lock()
THREAD_LOCK_PROCESS = threading.Lock()
UPSAMPLER_BG = None
UPSAMPLER = {"2":None, "4":None}
UPSAMPLER_FAST = None

def get_full_upsampler(up_by=4):
    #https://github.com/xinntao/Real-ESRGAN/blob/master/inference_realesrgan.py
    global UPSAMPLER
    if UPSAMPLER[str(up_by)] == None:
        with THREAD_LOCK_UPSAMPLER:
            if UPSAMPLER[str(up_by)] == None:
                bg_model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'RealESRGAN_x4plus.pth')
                face_model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'GFPGANv1.4.pth')
                upsampler = RealESRGANer(
                    scale=up_by,
                    model_path=bg_model_path,
                    dni_weight=None,
                    model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
                    tile=0,
                    tile_pad=10,
                    pre_pad=0,
                    half=False,
                    gpu_id=0
                )
                face_upsampler = GFPGANer(
                    model_path=face_model_path,
                    upscale=up_by,
                    arch='clean',
                    channel_multiplier=2,
                    bg_upsampler=upsampler
                )
                UPSAMPLER[str(up_by)] = face_upsampler
    return UPSAMPLER[str(up_by)]

def get_bg_upsampler(upscale=4):
    #https://github.com/xinntao/Real-ESRGAN/blob/master/inference_realesrgan.py
    global UPSAMPLER_BG
    if not UPSAMPLER_BG:
        with THREAD_LOCK_UPSAMPLER:
            if not UPSAMPLER_BG:
                bg_model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'RealESRGAN_x4plus.pth')
                upsampler = RealESRGANer(
                    scale=upscale,
                    model_path=bg_model_path,
                    dni_weight=None,
                    model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
                    tile=0,
                    tile_pad=10,
                    pre_pad=0,
                    half=False,
                    gpu_id=0
                )
                UPSAMPLER_BG = upsampler
    return UPSAMPLER_BG

def get_face_upsampler(upscale=2, bg_upsampler=None):
    #https://github.com/xinntao/Real-ESRGAN/blob/master/inference_realesrgan.py
    global UPSAMPLER_FAST
    if not UPSAMPLER_FAST:
        with THREAD_LOCK_UPSAMPLER_FAST:
            if not UPSAMPLER_FAST:
                face_model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'GFPGANv1.4.pth')
                face_upsampler = GFPGANer(
                    model_path=face_model_path,
                    upscale=upscale,
                    arch='clean',
                    channel_multiplier=2,
                    bg_upsampler=bg_upsampler
                )
                UPSAMPLER_FAST = face_upsampler
    return UPSAMPLER_FAST

def get_upsampler(fast=True, up_by=4):
    return get_face_upsampler(up_by) if fast else get_full_upsampler(up_by)

def upsample(image_data, fast=True, has_aligned=False, up_by=None):
    from util import get_face_analyser
    from masks import get_final_image, merge_original
    from insightface.utils import face_align

    if up_by == None and fast:
        up_by = 2
    elif up_by == None:
        up_by = 4

    #if fast:
    upsampler = get_upsampler(fast, up_by=up_by)

    with THREAD_LOCK_PROCESS:
        _, _, output = upsampler.enhance(image_data, has_aligned=has_aligned, only_center_face=False, paste_back=True)

        upsampler.cleanup() #Requires using https://github.com/postworthy/GFPGAN
    
    return output
    #else:
        #bg_upsampler = get_bg_upsampler(4)
        #face_upsampler = get_face_upsampler(4)
        #face_analyser = get_face_analyser()

        #with THREAD_LOCK_PROCESS:
        #    bg_img = bg_upsampler.enhance(image_data)[0]
        #    face = face_analyser.get(bg_img)[0]
        #    warped_img, estimated_norm = face_align.norm_crop2(bg_img, face.kps, 512)
        #    _, restored_faces, _ = face_upsampler.enhance(image_data, has_aligned=has_aligned, only_center_face=False, paste_back=False)
        #    #output = get_final_image([[restored_faces[0]]], bg_img, [[estimated_norm]])
        #    output = merge_original(bg_img, restored_faces[0], warped_img, estimated_norm)

        #    #Requires using https://github.com/postworthy/GFPGAN
        #    ##bg_upsampler.cleanup()
        #    face_upsampler.cleanup() 
            
        #return output

def upsample_batch(images:list, fast=True, has_aligned=False, up_by=None):
    from util import get_face_analyser
    from masks import get_final_image, merge_original

    if up_by is None and fast:
        up_by = 2
    elif up_by is None:
        up_by = 4

    upsampler = get_upsampler(fast, up_by=up_by)

    processed_images = []
    with THREAD_LOCK_PROCESS:
        for image_data in images:
            _, _, output = upsampler.enhance(image_data, has_aligned=has_aligned, only_center_face=False, paste_back=True)
            processed_images.append(output)
        
        upsampler.cleanup()  # Requires using https://github.com/postworthy/GFPGAN

    return processed_images




    return results