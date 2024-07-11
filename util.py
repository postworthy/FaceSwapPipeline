
import os
import cv2
os.environ['OMP_NUM_THREADS'] = '1'
import threading
import numpy as np
import torch
import insightface
import onnxruntime
from io import BytesIO
import imageio
from PIL import Image, ImageSequence
import onnx

global_vars = None

THREAD_LOCK_FACEANALYSER = threading.Lock()
THREAD_LOCK_FACESWAPPER = threading.Lock()

FACE_SWAPPER_ONNX_MODEL_AND_SESSION = None
FACE_SWAPPER = None
FACE_SWAPPER_GHOST = None
FACE_ANALYSER = None

#PROVIDERS = onnxruntime.get_available_providers()

#if 'TensorrtExecutionProvider' in PROVIDERS:
#    PROVIDERS.remove('TensorrtExecutionProvider')

#if 'AzureExecutionProvider' in PROVIDERS:
#    PROVIDERS.remove('AzureExecutionProvider')

PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider']



def get_global_vars():
    global global_vars
    return global_vars

def set_global_vars(var):
    global global_vars
    global_vars = var

def get_face_analyser():
    global FACE_ANALYSER
    if not FACE_ANALYSER:
        with THREAD_LOCK_FACEANALYSER:
            if not FACE_ANALYSER:
                FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=PROVIDERS)
                FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
                print("*** Face Analyser Loaded ***")
    return FACE_ANALYSER

def get_face_swapper():
    global FACE_SWAPPER
    if not FACE_SWAPPER:
        with THREAD_LOCK_FACESWAPPER:
            if not FACE_SWAPPER:
                FACE_SWAPPER = insightface.model_zoo.get_model('/root/.insightface/models/inswapper_128.onnx', providers=PROVIDERS)
                print("*** Face Swapper Loaded ***")
    return FACE_SWAPPER

def get_face_swapper_onnx_model_and_session():
    global FACE_SWAPPER_ONNX_MODEL_AND_SESSION
    if not FACE_SWAPPER_ONNX_MODEL_AND_SESSION:
        with THREAD_LOCK_FACESWAPPER:
            if not FACE_SWAPPER_ONNX_MODEL_AND_SESSION:
                model_file = '/root/.insightface/models/inswapper_128.onnx'
                model = onnx.load(model_file)
                session = onnxruntime.InferenceSession(model_file, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                FACE_SWAPPER_ONNX_MODEL_AND_SESSION = (model, session)
                print("*** Face Swapper Onnx Loaded ***")
    return FACE_SWAPPER_ONNX_MODEL_AND_SESSION

def get_faces(faces_directory='/app/faces/'):
    with torch.no_grad():
        face_analyser = get_face_analyser()
        face_swapper = get_face_swapper()

        for filename in os.listdir(faces_directory):
            if any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                image_path = os.path.join(faces_directory, filename)
                _, source_image = read_and_scale(image_path)
                source_face = min(face_analyser.get(source_image), key=lambda x: x.bbox[0])
                yield source_face

def select_random_image(faces_directory='/app/faces/'):
    import random
    # List all files in the directory
    files = os.listdir(faces_directory)
    
    # Filter out only image files and get their full paths
    image_files = [os.path.join(faces_directory, file) for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("No image files found in the directory.")
        return None

    # Sort files by creation time (most recent first)
    image_files.sort(key=os.path.getctime, reverse=True)

    # Keep only the last N files (or fewer if less than 10 files are present)
    last_10_files = image_files[:25]

    # Select a random image file from the last 10
    random_image = random.choice(last_10_files)
    
    # Return the full path of the image
    return random_image

def count_files_in_folder(directory):
    try:
        # List all entries in the directory
        entries = os.listdir(directory)

        # Count only files, excluding directories
        file_count = sum(1 for entry in entries if os.path.isfile(os.path.join(directory, entry)))
        
        return file_count
    except Exception as e:
        return 0

def read_and_scale(img_path, WxH=128):
    img = cv2.imread(img_path)
    return scale_image(img, WxH)

def scale_image(img, WxH=128):
    from util import get_face_analyser
    from upsampler import upsample

    face_analyser = get_face_analyser()
    
    height, width = img.shape[:2]

    if height < 256 or width < 256:
        img = upsample(img, False)
        #print(f'scale_image::upsample::img: {img.shape}')
        #Image.fromarray(cv2.cvtColor(img,cv2.COLOR_RGBA2BGR)).save('/app/faces/tmp/z.jpg')

    faces = face_analyser.get(img)

    scaling_factor = 1.0
    for bbox in [face.bbox for face in faces]:
        width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if width > WxH or height > WxH:
            scale =(WxH * 1.0) / max(width, height)
            scaling_factor = min(scaling_factor, scale)

    if scaling_factor < 1.0:
        new_size = (int(img.shape[1] * scaling_factor), int(img.shape[0] * scaling_factor))
        return scaling_factor, cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    else:
        return scaling_factor, img

def preprocess_image(image, noise_mean=0, noise_sigma=40):
    with torch.no_grad():
        face_analyser = get_face_analyser()

        faces = sorted(face_analyser.get(np.array(image)), key=lambda x: x.bbox[0])

        noisy_image = image + np.random.normal(noise_mean, noise_sigma, np.array(image).shape)
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        noisy_faces = sorted(face_analyser.get(np.array(noisy_image)), key=lambda x: x.bbox[0])
        
        return {
            "faces": faces,
            "first_face": faces[0] if len(faces) > 0 else None,
            "noisy_faces": noisy_faces,
            "first_noisy_face": noisy_faces[0] if len(noisy_faces) > 0 else None,
            "data": image,
            "noisy_data": noisy_image
        }

def process_image_alt2(image, source, use_noisy_images = False): 
    from new_faces import single_swap
    frame = single_swap(
        np.array(image["data"]), 
        image["first_face"] if not use_noisy_images else image["first_noisy_face"], 
        source["first_face"] if not use_noisy_images else source["first_noisy_face"],
        upsample_level=1)
    return Image.fromarray(frame)

def process_image_alt(image, source, use_noisy_images = False): 
    with torch.no_grad():
        face_swapper = get_face_swapper()

        frame = face_swapper.get(np.array(image["data"]), 
                                 image["first_face"] if not use_noisy_images else image["first_noisy_face"], 
                                 source["first_face"] if not use_noisy_images else source["first_noisy_face"], 
                                 paste_back=True)

        return Image.fromarray(frame)

def process_image(image, faces_directory='/app/faces/'):  
    with torch.no_grad():
        face_analyser = get_face_analyser()
        face_swapper = get_face_swapper()
        
        for filename in os.listdir(faces_directory):
            if any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                image_path = os.path.join(faces_directory, filename)
                _, source_image = read_and_scale(image_path)
                source_face = min(face_analyser.get(source_image), key=lambda x: x.bbox[0])
                #print(f'Face Dimensions: {(source_face.bbox[2] - source_face.bbox[0])}x{(source_face.bbox[3] - source_face.bbox[1])}')
                frame = np.array(image)
                faces = sorted(face_analyser.get(np.array(frame)), key=lambda x: x.bbox[0])
            
                if faces:
                    for i, face in enumerate(faces):
                        frame = face_swapper.get(frame, face, source_face, paste_back=True)
                
                result = frame
                yield Image.fromarray(result)

def process_image_with_ghost(image, faces_directory='/app/faces/', upscale=False):  
    from ghost import ghost_process_image

    for filename in os.listdir(faces_directory):
        if any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
            image_path = os.path.join(faces_directory, filename)
            #_, source_image = read_and_scale(image_path, 256)
            source_image = cv2.imread(image_path)
            frame = np.array(image)

            result = ghost_process_image(frame, source_image, upscale)
            yield Image.fromarray(result)

def batch_process_image(into_images, from_image, upscale=False):
    from util import get_face_analyser, get_face_swapper
    from upsampler import upsample_batch

    def process_batch(into_images, from_image):
        face_analyser = get_face_analyser()
        face_swapper = get_face_swapper()

        batch_has_face = []

        # Process the source (from_image) to find the source face
        from_faces = face_analyser.get(from_image)
        from_faces_sorted = sorted(from_faces, key=lambda x: x.bbox[0])
        if from_faces_sorted:
            source_face = from_faces_sorted[0]

            # Process each image in the batch
            for into_image in into_images:
                faces = face_analyser.get(into_image)
                if faces:
                    face = faces[0]  # Assuming only one face per image for simplicity
                    into_image = face_swapper.get(into_image, face, source_face, paste_back=True)
                    batch_has_face.append(into_image)
                else:
                    batch_has_face.append(into_image)
        else:
            # If no face is found in the source image, return the original images
            batch_has_face = into_images

        return batch_has_face

    # Convert into_images and from_image to the necessary format
    into_images = [img[:, :, ::-1] for img in into_images]  # Convert list of into images
    from_image = from_image[:, :, ::-1]  # Convert single from image to batch format

    # Process images in batch
    processed_images = process_batch(into_images, from_image)

    if upscale:
        processed_images = upsample_batch(processed_images)
        processed_images = [cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)) for img in processed_images]


    # Convert images back to original format
    #processed_images = [img[:, :, ::-1] for img in processed_images]

    return processed_images


def export_as_gif(frames, fps):
    numpy_frames = [np.array(frame.convert('RGB')) for frame in frames]
    # Duplicate frames in reverse order
    numpy_frames += numpy_frames[::-1]
    gif_output = BytesIO()
    imageio.mimsave(gif_output, numpy_frames, format='gif', duration=1000/fps, loop=0, opt_level=3)
    gif_output.seek(0)

    return gif_output

def create_new_directory(base_directory):
    if not os.path.exists(base_directory):
        raise ValueError(f"Base directory does not exist: {base_directory}")

    dir_count = sum(os.path.isdir(os.path.join(base_directory, d)) for d in os.listdir(base_directory))
    new_dir_name = f"{dir_count + 1}"
    new_dir_path = os.path.join(base_directory, new_dir_name)

    os.mkdir(new_dir_path)

    return new_dir_path