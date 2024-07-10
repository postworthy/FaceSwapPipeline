import torch
import onnx
import onnxruntime
from onnx import numpy_helper
import numpy as np
from util import get_face_analyser
from insightface.utils import face_align
#from upsampler import upsample_batch
import os
#from util import read_and_scale
import torch.nn.functional as F
import cv2
from masks import merge_original, merge_original_with_mask
#from masks import face_mask_static
from typing import Callable, List
import torchvision.transforms as transforms
from PIL import Image
from network.AEI_Net import AEI_Net
from network.iresnet import iresnet100
import insightface
import threading

GLOBAL_MODEL = None
THREAD_LOCK_MODEL = threading.Lock()
IS_HALF = True
def save_img_fromarray(np_array, path):

    if np_array.dtype == np.float32:
        np_array = np_array * 255  # Scale from 0-1 to 0-255
        np_array = np.clip(np_array, 0, 255)  # Ensure all values are within 0-255
        np_array = np_array.astype(np.uint8)  # Convert to unsigned byte format

    # Create the image object, ensuring it's in 'L' mode for grayscale (if single-channel)
    if np_array.ndim == 2:  # It's a single-channel image
        image = Image.fromarray(np_array, 'L')
    elif np_array.shape[2] == 3:  # It's a three-channel image
        image = Image.fromarray(np_array, 'RGB')
    else:
        raise ValueError("The array must be either 2-dimensional or have 3 channels")



def load_model(model_file='./G_latest.pth', mask_model_file='./G_latest_mask.pth', arcface_file='./backbone.pth'):
    # main model for generation
    G = AEI_Net('unet', num_blocks=2, c_id=512).cuda()
    G.train() 
    #G.eval()
    result = G.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
    print(result)

    G_mask = AEI_Net('unet', num_blocks=2, c_id=512).cuda()
    #G_mask.train()
    G_mask.eval()
    result = G_mask.load_state_dict(torch.load(mask_model_file, map_location=torch.device('cpu')))
    print(result)

    if IS_HALF:
        G = G.half()
        G_mask = G_mask.half()

    # arcface model to get face embedding
    netArc = iresnet100(fp16=False)
    result = netArc.load_state_dict(torch.load(arcface_file))
    print(result)
    netArc=netArc.cuda()
    netArc.eval()
    input_size = (256, 256)
    return {
        "model":        G,
        "mask":         G_mask,
        "arcface":      netArc,
        "input_size":   input_size,
    }

def get_model():
    global GLOBAL_MODEL
    global THREAD_LOCK_MODEL
    if not GLOBAL_MODEL:
        with THREAD_LOCK_MODEL:
            if not GLOBAL_MODEL:
                GLOBAL_MODEL = load_model()
                print("*** Ghost Model Loaded ***")
    return GLOBAL_MODEL

def normalize_and_torch(input: np.ndarray) -> torch.tensor:
    if input.ndim == 3:
        image = input
        image = torch.tensor(image.copy(), dtype=torch.float32).cuda()
        if image.max() > 1.:
            image = image/255.
        image = image.permute(2, 0, 1).unsqueeze(0)
        image = (image - 0.5) / 0.5
        return image
    elif input.ndim == 4:
        return normalize_and_torch_batch(input)
    else:
        raise ValueError(f"Input must be either a single image or a batch of images. shape: {input.shape} ndim: {input.ndim}")

def normalize_and_torch_batch(frames: np.ndarray) -> torch.tensor:
    """
    Normalize batch images and transform to torch
    """
    batch_frames = torch.from_numpy(frames.copy()).cuda()
    if batch_frames.max() > 1.:
        batch_frames = batch_frames/255.
    
    batch_frames = batch_frames.permute(0, 3, 1, 2)
    batch_frames = (batch_frames - 0.5)/0.5

    return batch_frames


def no_normalize_and_torch(image: np.ndarray) -> torch.tensor:
    image = torch.tensor(image.copy(), dtype=torch.float32).cuda()
    image = image.permute(2, 0, 1).unsqueeze(0)
    return image

def torch2image(torch_image: torch.tensor) -> np.ndarray:
    batch = False
    if torch_image.dim() == 4:
        torch_image = torch_image[:8]
        batch = True
    device = torch_image.device

    denorm_image = torch_image
    if batch:
        denorm_image = denorm_image.permute(0, 2, 3, 1)
    else:
        denorm_image = denorm_image.permute(1, 2, 0)
    np_image = denorm_image.detach().cpu().numpy()
    np_image = np.clip(np_image*255., 0, 255).astype(np.uint8)
    
    return np_image

def get_mask(G_mask, netArc, target_image_256, target_image_224):
    #G_mask.train()  # Set the model to train mode (eval doesn't seem to provide the correct results)
    #netArc.eval()

    global THREAD_LOCK_MODEL

    with torch.no_grad():
        target_tensor_256 = normalize_and_torch(target_image_256)
        target_tensor_224 = normalize_and_torch(target_image_224)

        embed = netArc(F.interpolate(target_tensor_224, scale_factor=0.5, mode='area'))
        
        if IS_HALF:
            target_tensor_256 = target_tensor_256.half()
            embed = embed.half()
        
        with THREAD_LOCK_MODEL:
            output_tensor, _ = G_mask(target_tensor_256, embed)
            
            # Assuming all channels must be >= 0.7 for the mask to be white
            mask = torch.all(output_tensor >= 0.7, dim=1, keepdim=True)  # Check if all channels >= 0.7
            mask = mask.expand(-1, 3, -1, -1)  # Expand back to have three channels for compatibility

            # Convert to numpy for final adjustments
            mask_np = mask.cpu().numpy().astype(float)  # Convert to float for compatibility with np.clip
            mask_np = np.squeeze(mask_np)  # Remove the single-dimensional entry from the shape
            mask_np = np.transpose(mask_np, (1, 2, 0))  # Reorder dimensions to Height x Width x Channels
            mask_np = np.clip(mask_np, 0, 1)  # Ensure all values are between 0 and 1
            mask_image = (mask_np * 255).astype(np.uint8)  # Convert to an unsigned byte format
            mask_image = np.mean(mask_image, axis=2)  # Convert to single-channel by averaging across color channels
        
        return mask_image

def get_mask_batch(G_mask, netArc, target_images_256, target_images_224):
    #G_mask.train()  # Set the model to train mode
    #netArc.eval()

    global THREAD_LOCK_MODEL

    with torch.no_grad():
        # Normalize and prepare batch tensors
        target_tensors_256 = normalize_and_torch(target_images_256)
        target_tensors_224 = normalize_and_torch(target_images_224)

        # Generate embeddings
        embeds = netArc(F.interpolate(target_tensors_224, scale_factor=0.5, mode='area'))
        
        if IS_HALF:
            target_tensors_256 = target_tensors_256.half()
            embeds = embeds.half()
        
        # Generate output masks
        with THREAD_LOCK_MODEL:
            output_tensors, _ = G_mask(target_tensors_256, embeds)
        
            # Create masks based on channels conditions
            masks = torch.all(output_tensors >= 0.7, dim=1, keepdim=True)  # Check if all channels >= 0.7
            masks = masks.expand(-1, 3, -1, -1)  # Expand back to three channels

            # Convert to numpy arrays
            masks_np = masks.cpu().numpy().astype(float)  # Convert to float for compatibility with np.clip
            # Process each item in the batch
            mask_images = []
            for i in range(masks_np.shape[0]):
                mask_single = np.squeeze(masks_np[i])  # Remove the single-dimensional entry
                mask_single = np.transpose(mask_single, (1, 2, 0))  # Reorder dimensions to Height x Width x Channels
                mask_single = np.clip(mask_single, 0, 1)  # Ensure values are between 0 and 1
                mask_single = (mask_single * 255).astype(np.uint8)  # Convert to unsigned byte format
                mask_single = np.mean(mask_single, axis=2)  # Convert to single-channel by averaging across color channels
                mask_images.append(mask_single)
            
        return mask_images

def perform_inference(G, netArc, target_image, source_image):
    #G.train()  # Set the model to train mode (eval doesn't seem to provide the correct results)
    #netArc.eval()

    global THREAD_LOCK_MODEL

    with torch.no_grad():
        target_tensor = normalize_and_torch(target_image)
        source_tensor = normalize_and_torch(source_image)

        embed = netArc(F.interpolate(source_tensor, scale_factor=0.5, mode='area'))
        
        if IS_HALF:
            target_tensor = target_tensor.half()
            embed = embed.half()
        
        with THREAD_LOCK_MODEL:
            output_tensor, _ = G(target_tensor, embed)
        
            #print(f"output_tensor shape: {output_tensor.shape}")
        
            output_image = torch2image(output_tensor)
            #output_image = adjust_hsv(output_image, calculate_hsv(target_image))
            #output_image = output_image[:,:,::-1] # Flip from BGR to RGB

        return output_image
    
def perform_inference2(G, target_image, source_embed):
    #G.train()  # Set the model to train mode (eval doesn't seem to provide the correct results)

    global THREAD_LOCK_MODEL

    with torch.no_grad():
        target_tensor = normalize_and_torch(target_image)

        embed = source_embed
        
        if IS_HALF:
            target_tensor = target_tensor.half()
            embed = embed.half()

        with THREAD_LOCK_MODEL:
            output_tensor, _ = G(target_tensor, embed)
            output_image = torch2image(output_tensor)
            #output_image = adjust_hsv(output_image, calculate_hsv(target_image))
            #output_image = output_image[:,:,::-1] # Flip from BGR to RGB

        return output_image
    
def perform_inference3(G, netArc, target_image, source_image, steps=5):
    #G.train()  # Set the model to train mode (eval doesn't seem to provide the correct results)
    if IS_HALF:
        G = G.half()

    with torch.no_grad():
        target_tensor = normalize_and_torch(target_image)

        target_image_resized = cv2.resize(target_image, (224, 224))

        #does stepping up result in better final images?
        embedings = blend_embeddings(netArc, [source_image, target_image_resized], steps)

        for i, embed in enumerate(embedings):
            if IS_HALF and target_tensor.dtype != torch.float16:
                target_tensor = target_tensor.half()
            if IS_HALF and embed.dtype != torch.float16:
                embed = embed.half()
            output_tensor, _ = G(target_tensor, embed)
            if IS_HALF and output_tensor.dtype != torch.float16:
                output_tensor = output_tensor.half()
            target_tensor = output_tensor

        output_image = torch2image(output_tensor)

        return output_image

def average_embeds(netArc, source_images):
    embeds = []
    for i, source_image in enumerate(source_images):
        source_tensor = normalize_and_torch(source_image)
        embed = netArc(F.interpolate(source_tensor, scale_factor=0.5, mode='area'))
        embeds.append(embed)
    stacked_embeds = torch.stack(embeds)  # This will have shape [N, 1, 512]
    average_embed = torch.mean(stacked_embeds, dim=0)  # Compute the mean across the first dimension
    return [average_embed]

def average_embeds_randomize(netArc, source_images):
    embeds = []
    num_images = len(source_images)
    
    # Generate random scaling factors and normalize them
    scaling_factors = np.random.rand(num_images)  # Create a random array of scaling factors
    scaling_factors /= scaling_factors.sum()  # Normalize so their sum is 1
    
    for i, source_image in enumerate(source_images):
        source_tensor = normalize_and_torch(source_image)
        embed = netArc(F.interpolate(source_tensor, scale_factor=0.5, mode='area'))
        
        # Apply scaling factor to each embed
        scaled_embed = embed * scaling_factors[i]
        embeds.append(scaled_embed)
    
    # Sum all the scaled embeddings
    total_embed = torch.stack(embeds).sum(dim=0)  # Sum across the embeddings, resulting in one average embedding

    return [total_embed]

def blend_embeddings(netArc, source_images, steps=100):
    # Assume source_images has exactly two embeddings
    embeds = []
    for source_image in source_images:
        source_tensor = normalize_and_torch(source_image)
        embed = netArc(F.interpolate(source_tensor, scale_factor=0.5, mode='area'))
        embeds.append(embed.squeeze(0))  # Remove batch dimension assuming it's 1, resulting in shape [512]
    
    # Generate blended embeddings
    blended_embeds = []
    for alpha in range(steps+1):  # Including both 0% and 100%
        blend_factor = alpha / (steps*1.0)
        blended_embed = (1 - blend_factor) * embeds[0] + blend_factor * embeds[1]
        blended_embeds.append(blended_embed.unsqueeze(0))  # Add batch dimension back for consistency
    
    return blended_embeds


def calculate_hsv(img):
    # Convert to HSV for brightness and saturation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)


    # Calculate average brightness and saturation
    avg_brightness = np.mean(v)
    avg_saturation = np.mean(s)

    # Calculate contrast (standard deviation of pixel intensities)
    avg_contrast = np.std(img)
    #avg_contrast = np.std(v)


    return avg_brightness, avg_contrast, avg_saturation

def adjust_hsv(img, target_hsv):
    target_brightness, target_contrast, target_saturation = target_hsv
    # Convert to HSV for saturation adjustment
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Adjust saturation
    s = np.clip((s * target_saturation) / np.mean(s), 0, 255).astype(hsv.dtype)

    # Adjust brightness and contrast
    if target_contrast > 0:
        alpha = target_contrast / np.std(v)
        v = np.clip((v - np.mean(v)) * alpha + target_brightness, 0, 255).astype(hsv.dtype)
    else:
        v = np.clip(v + target_brightness - np.mean(v), 0, 255).astype(hsv.dtype)

    # Merge back and convert to BGR
    adjusted_img = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    return adjusted_img


def get_ghost_face_swapper():
    ghost_model = load_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #perform_inference(ghost_model["model"], ghost_model["arcface"], into_img, from_img, device)

    return {
        "get": lambda into_img, from_img: perform_inference(ghost_model["model"], ghost_model["arcface"], into_img, from_img, device),
    }

def ghost_process_image(into_image, from_image, upscale=False):
    from upsampler import upsample_batch, upsample
    from util import get_face_analyser    
    
    into_image = into_image[:, :, ::-1] # This assimes cv2.imread which reads in as BGR but we pass in RGB
    from_image = from_image[:, :, ::-1] # This assimes cv2.imread which reads in as BGR but we pass in RGB


    ghost_model = get_model()
    face_analyser = get_face_analyser()
    
    into_faces = face_analyser.get(into_image)
    final_result = into_image
    for i, into_face in enumerate(into_faces):
        into_img, estimated_norm = face_align.norm_crop2(into_image, into_face.kps, 256)
        into_img_mask, estimated_norm_mask = face_align.norm_crop2(into_image, into_face.kps, 224)
        into_img = into_img.astype(np.float32)
        #into_img = (into_img - mean) / std
        from_faces = sorted(face_analyser.get(from_image), key=lambda x: x.bbox[0])
        if from_faces and len(from_faces) > 0:
            from_face = from_faces[0]
            from_img, _ = face_align.norm_crop2(from_image, from_face.kps, 224)
            from_img = from_img.astype(np.float32)
            #from_img = (from_img - mean) / std
            output_image = perform_inference(ghost_model["model"], ghost_model["arcface"], into_img, from_img)
            mask_image = get_mask(ghost_model["mask"], ghost_model["arcface"], into_img, into_img_mask)
            rgb_fake = np.array(output_image)
            into_img = np.clip(into_img*255., 0, 255).astype(np.uint8)
            #save_img_fromarray(into_img, f'/app/output/into_face_{i}.jpg')
            #save_img_fromarray(rgb_fake, f'/app/output/out_face_{i}.jpg')
            #save_img_fromarray(mask_image, f'/app/output/mask_face_{i}.jpg')
            final_result = merge_original_with_mask(final_result, rgb_fake, into_img, estimated_norm, mask_image, mask_border_thickness=5)
            
    if upscale:
        final_result = upsample(final_result, False, up_by=2)                

    return final_result

def ghost_batch_process_image(into_images, from_image, upscale=False):
    from upsampler import upsample_batch
    from util import get_face_analyser

    into_images = [img[:, :, ::-1] for img in into_images]  # Convert list of into images
    from_image = from_image[:, :, ::-1]  # Convert single from image to batch format

    ghost_model = get_model()
    face_analyser = get_face_analyser()

    #print(f"batch size: {len(into_images)}")

    batch_into_faces = []
    batch_into_masks = []
    batch_into_norms = []
    batch_has_face   = []
    for i, into_image in enumerate(into_images):
        try:
            face = face_analyser.get(into_image)[0]
        except:
            face=None
            print(f"No face at {i}")

        if face:
            into_img, estimated_norm = face_align.norm_crop2(into_image, face.kps, 256)
            into_img_mask, _ = face_align.norm_crop2(into_image, face.kps, 224)
            into_img = into_img.astype(np.float32)
            batch_into_faces.append(into_img)
            batch_into_masks.append(into_img_mask)
            batch_into_norms.append(estimated_norm)
            batch_has_face.append(True)
        else:
            batch_has_face.append(False)

    from_faces = face_analyser.get(from_image)
    from_faces_batch = []

    from_faces_sorted = sorted(from_faces, key=lambda x: x.bbox[0])
    if from_faces_sorted:
        from_face = from_faces_sorted[0]

        for _ in batch_into_faces:
            from_img, _ = face_align.norm_crop2(from_image, from_face.kps, 224)
            from_img = from_img.astype(np.float32)
            from_faces_batch.append(from_img)

    if len(batch_into_faces) > 0:
        batch_into_faces = np.array(batch_into_faces)
        from_faces_batch = np.array(from_faces_batch)
        batch_into_masks = np.array(batch_into_masks)
        
        
        #print(batch_into_faces.shape)
        #print(from_faces_batch.shape)
        #print(batch_into_masks.shape)

        output_images = perform_inference(ghost_model["model"], ghost_model["arcface"], batch_into_faces, from_faces_batch)
        mask_images = get_mask_batch(ghost_model["mask"], ghost_model["arcface"], batch_into_faces, batch_into_masks)
    else:
        output_images = []
        mask_images = []

    final_results = []
    no_face_offset = 0
    for i, has_face in enumerate(batch_has_face):
        if has_face:
            rgb_fake = output_images[i+no_face_offset]
            mask_image = mask_images[i+no_face_offset]
            estimated_norm = batch_into_norms[i+no_face_offset]
            into_img = np.clip(batch_into_faces[i+no_face_offset]*255., 0, 255).astype(np.uint8)
            #print(f"rgb_fake shape: {rgb_fake.shape}")  # Should be (256, 256, 3)
            #print(f"into_img shape: {into_img.shape}")  # Should be (256, 256, 3)
            final_result = merge_original_with_mask(into_images[i], rgb_fake, into_img, estimated_norm, mask_image)
        else:
            no_face_offset = no_face_offset - 1
            final_result = into_images[i]
        
        final_results.append(final_result)

    return final_results

def g(randomize=False, upscale=True, average=False):
    from util import read_and_scale, preprocess_image, export_as_gif
    from new_faces import faces_from_image_gen
    from upsampler import upsample_batch, upsample
    import datetime, random; 
    prefix = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}_"

    ghost_model = get_model()

    # Perform inference on images
    output_directory='/app/faces/tmp'
    faces_directory='/app/faces/'
    
    from_images = [os.path.join(faces_directory, file) for file in os.listdir(faces_directory) if '_target_' in file and file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not randomize:
        into_images=[]
        temp = [os.path.join(faces_directory, file) for file in os.listdir(faces_directory) if '_target_' not in file and file.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for _, f in enumerate(temp):
            _, x = read_and_scale(f, 256)
            into_images.append(preprocess_image(x))
    else:
        into_images = faces_from_image_gen(["Beautiful Actress Emma Watson"], 256, None)

    face_analyser = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_analyser.prepare(ctx_id=0, det_size=(640, 640))

    if average:
        all_from_images=[]
        for _, from_img_path in enumerate(from_images):
            _, from_image = read_and_scale(from_img_path, 224)
            from_face = face_analyser.get(from_image)[0]
            if from_face:
                from_img, _ = face_align.norm_crop2(from_image, from_face.kps, 224)
                all_from_images.append(from_img)
        #average_embeds = average_embeds_randomize(ghost_model["arcface"], all_from_images)
        average_embeds = blend_embeddings(ghost_model["arcface"], all_from_images)

    for i, into_img_ in enumerate(into_images):
        into_image = np.array(into_img_["data"])
        into_face = into_img_["first_face"]
        if into_face:
            into_img, estimated_norm = face_align.norm_crop2(into_image, into_face.kps, 256)
            into_img = into_img
            output_images = []
            if average:                
                for i, average_embed in enumerate(average_embeds):
                    output_image = perform_inference2(ghost_model["model"], into_img, average_embed)
                    if i == 0:
                        into_img = output_image
                    elif i > 2:
                        output_images.append(output_image)
            else:
                for _, from_img_path in enumerate(from_images):
                    _, from_image = read_and_scale(from_img_path, 224) #cv2.imread(from_img_path)
                    from_face = face_analyser.get(from_image)[0]
                    if from_face:
                        from_img, _ = face_align.norm_crop2(from_image, from_face.kps, 224)
                        from_img = from_img
                        output_image = perform_inference(ghost_model["model"], ghost_model["arcface"], into_img, from_img)
                        output_images.append(output_image)
                        output_image = perform_inference3(ghost_model["model"], ghost_model["arcface"], into_img, from_img, 5)
                        output_images.append(output_image)

            gif_images = []
            for j, output_image in enumerate(output_images):
                bgr_fake = np.array(output_image)
                #bgr_fake = adjust_hsv(bgr_fake, calculate_hsv(into_img))
                #into_image = adjust_hsv(into_image, avg_hsv)
                merged = merge_original(into_image, bgr_fake, into_img, estimated_norm)
                if upscale:
                    final_image = upsample(merged, False, up_by=2)
                    final_image = Image.fromarray(final_image)
                    gif_images.append(final_image)
                    final_image.save(f'{output_directory}/{prefix}merged_{i}_{j}.jpg')
                else:
                    merged = Image.fromarray(merged)
                    gif_images.append(merged)
                    merged.save(f'{output_directory}/{prefix}merged_{i}_{j}.jpg')

            if average:
                gif_bytes = export_as_gif(gif_images, 20)
                output_path = f"/app/output/{prefix}img_{i}.gif"
                with open(output_path, "wb") as f:
                    f.write(gif_bytes.getvalue())
                    