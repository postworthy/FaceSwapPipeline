import uuid
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, AutoencoderKL, DDPMScheduler, DPMSolverMultistepScheduler
from insightface.app import FaceAnalysis
import numpy as np
import cv2
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
from torchvision import transforms
from util import get_face_analyser, get_face_swapper, read_and_scale

swapped=False

# Your face detection and swapping functions
def detect_face(image):
    face_analyser = get_face_analyser()
    faces = face_analyser.get(image)
    if faces:
        print("Face detected.")
        return faces[0]
    return None

def swap_face_in_image(image, source_image):
    with torch.no_grad():
        face_analyser = get_face_analyser()
        face_swapper = get_face_swapper()

        source_face = min(face_analyser.get(source_image), key=lambda x: x.bbox[0])
        frame = np.array(image)
        faces = sorted(face_analyser.get(np.array(frame)), key=lambda x: x.bbox[0])
    
        if faces:
            for i, face in enumerate(faces):
                frame = face_swapper.get(frame, face, source_face, paste_back=True)
                print(frame.shape)
        return Image.fromarray(frame)

def init_sdxl():
    
    pipelines = []
    pipelines.append(
        StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, 
            variant="fp16", 
            use_safetensors=True, 
        )
    )
    #scheduler = EulerAncestralDiscreteScheduler.from_config(pipelines[0].scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing")
    #scheduler = DDPMScheduler.from_config(pipelines[0].scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing")
    scheduler = DPMSolverMultistepScheduler.from_config(pipelines[0].scheduler.config, use_karras_sigmas=True)
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, use_safetensors=True)
    for pipeline in pipelines:
        pipeline.scheduler = scheduler
        pipeline.vae = vae
        pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead")
        pipeline.unet.enable_xformers_memory_efficient_attention()
        #pipeline.enable_model_cpu_offload()
        pipeline.to("cuda" if torch.cuda.is_available() else "cpu")


    return pipelines[0]

def latents_to_rgb(latents):
    weights = (
        (60, -60, 25, -70),
        (60,  -5, 15, -50),
        (60,  10, -5, -35)
    )

    weights_tensor = torch.t(torch.tensor(weights, dtype=latents.dtype).to(latents.device))
    biases_tensor = torch.tensor((150, 140, 130), dtype=latents.dtype).to(latents.device)
    rgb_tensor = torch.einsum("...lxy,lr -> ...rxy", latents, weights_tensor) + biases_tensor.unsqueeze(-1).unsqueeze(-1)
    image_array = rgb_tensor.clamp(0, 255)[0].byte().cpu().numpy()
    image_array = image_array.transpose(1, 2, 0)

    return Image.fromarray(image_array)

def latents_to_rgba(latents):
    weights = (
        (60, -60, 25, -70),
        (60,  -5, 15, -50),
        (60,  10, -5, -35),
        (20,  30, -10, 40)
    )
    biases = (150, 140, 130, 120)

    weights_tensor = torch.tensor(weights, dtype=latents.dtype, device=latents.device).t()
    
    biases_tensor = torch.tensor(biases, dtype=latents.dtype, device=latents.device)
    rgba_tensor = torch.einsum("...lxy,lr -> ...rxy", latents, weights_tensor) + biases_tensor.view(-1, 1, 1)
    
    image_array = rgba_tensor.clamp(0, 255)[0].byte().cpu().numpy()
    image_array = image_array.transpose(1, 2, 0)

    return Image.fromarray(image_array)

def rgba_to_latents(rgba_tensor):
    weights = (
        (60, -60, 25, -70),
        (60,  -5, 15, -50),
        (60,  10, -5, -35),
        (20,  30, -10, 40)
    )
    biases = (150, 140, 130, 120)

    # Convert weights and biases to tensors
    weights_tensor = torch.tensor(weights, dtype=rgba_tensor.dtype, device=rgba_tensor.device).t()
    biases_tensor = torch.tensor(biases, dtype=rgba_tensor.dtype, device=rgba_tensor.device)

    # Remove biases
    rgba_bias_removed = rgba_tensor - biases_tensor.view(-1, 1, 1)

    # Compute the inverse of the weights tensor
    weights_inv = torch.inverse(weights_tensor)

    # Perform the reverse transformation
    latents = torch.einsum('...rxy, rl -> ...lxy', rgba_bias_removed, weights_inv)

    return latents


# Callback function for face-swapping at each step
def face_swap_callback(pipeline, step, timestep, callback_kwargs):
    global swapped

    latents_output = callback_kwargs["latents"]
    latents = callback_kwargs["latents"]

    # Decode latents to an image
    #vae = pipeline.vae
    #latents = latents / (vae.config.scaling_factor if 'scaling_factor' in vae.config else 0.18215)
    #decoded_image = vae.decode(latents).sample
    ##decoded_image = (decoded_image / 2 + 0.5).clamp(0, 1)
    #decoded_image = decoded_image.to(torch.float32).detach().cpu().numpy()
    #image_pil = Image.fromarray(np.uint8(decoded_image[0].transpose(1, 2, 0) * 255))
    image_pil = latents_to_rgba(latents)
    random_id = uuid.uuid4()  # Generate a random UUID
    image_pil.save(f"/app/output/callback_step_{step}_{timestep}_{random_id}.png")
    alpha_channel = image_pil.getchannel('A')
    image_np = np.array(image_pil.convert('RGB'))

    if swapped:
        return {"latents": latents}  

    # Detect and swap face if detected
    detected_face = detect_face(image_np)
    if detected_face:
        # Load the target face image
        _, source_face_img = read_and_scale("/app/faces/1.jpg")
        
        image_pil = swap_face_in_image(image_np, source_face_img)
        image_pil = image_pil.convert('RGBA')
        image_pil.putalpha(alpha_channel)
        image_pil.save(f"/app/output/callback_step_swap_{step}_{timestep}_{random_id}.png")

        
        #preprocess = transforms.Compose([
        #    transforms.ToTensor(),  # Convert PIL image to Tensor
        #    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        #])
        #rgba_tensor = preprocess(image_pil).unsqueeze(0).to(latents.device).to(dtype=pipeline.vae.dtype)
        #latents = rgba_to_latents(rgba_tensor)
        # Re-encode the swapped image back to latents
        preprocess = transforms.Compose([
            transforms.ToTensor(),  # Convert PIL image to Tensor
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])
        image_tensor = preprocess(image_pil).unsqueeze(0).to(latents.device)  # Add batch dimension
        image_tensor = image_tensor.to(dtype=pipeline.vae.dtype)  # Cast to BFloat16 if needed
        vae_output = pipeline.vae.encode(image_tensor)
        latents = vae_output.latent_dist.sample() * pipeline.vae.config.scaling_factor
        latents_output = latents
        #latents_output[0:1] = latents
        #latents_output[1:2] = latents
        
        swapped=True
    
    
    # Update latents in the callback
    return {"latents": latents_output}

# Example Usage
if __name__ == "__main__":
    # Load your base model
    pipeline = init_sdxl()
    target_face_image = "/app/faces/1.jpg"

    image = pipeline(
        prompt="a photo of a handsome man",
        num_inference_steps=50,
        callback_on_step_end=face_swap_callback,  # Register the face-swap callback
        callback_on_step_end_tensor_inputs=["latents"],  # Ensure 'latents' are passed to the callback
    ).images[0]

    random_id = uuid.uuid4()  # Generate a random UUID
    image.save(f"/app/output/output_image_{random_id}.png")
    print("Inference completed and image saved.")