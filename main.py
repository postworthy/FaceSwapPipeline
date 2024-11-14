import os
os.environ['OMP_NUM_THREADS'] = '1'
from diffusers import DiffusionPipeline, AutoencoderKL, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, EulerAncestralDiscreteScheduler, DDPMScheduler, DPMSolverMultistepScheduler, AutoPipelineForText2Image, StableDiffusion3Pipeline, FluxPipeline
import torch
import torch._dynamo
import random
import sys
import code
from util import process_image, process_image_with_ghost, export_as_gif, set_global_vars, get_global_vars
from new_faces import face_permutations, swap_all_in_one
from PIL.ExifTags import TAGS
import json
import piexif
from PIL import Image
import numpy as np

from optimum.quanto import freeze, qfloat8, quantize
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer,T5EncoderModel, T5TokenizerFast

#https://huggingface.co/docs/diffusers/main/en/optimization/fp16
torch.backends.cuda.matmul.allow_tf32 = True

torch.jit.enable_onednn_fusion(True)
torch.backends.cudnn.benchmark = True

torch._dynamo.config.suppress_errors = True

n_steps = 50
high_noise_frac = 0.8
refiner_strength = 0.003
num_images_per_prompt=2
prompt = "radiant colorful  80s retro t-shirt design graphic of a synth wave (((husband and wife))) (bust) looking at camera in front of a minimalist sunset with a contour and dark background"
negative_prompt = "pink"
height = 1024
width = 1024

flux=None
sd3=None
base = None
refiner = None
turbo = None

needs_init_flux = True
needs_init_sd3 = True
needs_init_sdxl = True
needs_init_turbo = True

def init(pipeline_name="sdxl"):
    if pipeline_name == "turbo":
        return init_turbo(), None
    elif pipeline_name == "sdxl":
        return init_sdxl()
    elif pipeline_name == "sd3":
        return init_sd3(), None
    else:
        return init_flux(), None

def save_quantized_model(model, path):
    torch.save(model.state_dict(), path)

def load_quantized_model(model_class, path):
    model = model_class()
    model.load_state_dict(torch.load(path))
    return model

def quantize_and_save(model, quantization_func, model_path):
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        quantization_func(model, weights=qfloat8)
        torch.save(model.state_dict(), model_path)


def init_flux():
    global flux, needs_init_flux

    needs_init_flux = False
    
    #https://gist.github.com/AmericanPresidentJimmyCarter/873985638e1f3541ba8b00137e7dacd9

    dtype = torch.bfloat16
    bfl_repo = "black-forest-labs/FLUX.1-schnell"
    revision = "refs/pr/1"

    print("Loading Flux Components")

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(bfl_repo, subfolder="scheduler", revision=revision)
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
    text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype, revision=revision)
    tokenizer_2 = T5TokenizerFast.from_pretrained(bfl_repo, subfolder="tokenizer_2", torch_dtype=dtype, revision=revision)
    vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder="vae", torch_dtype=dtype, revision=revision)
    transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer", torch_dtype=dtype, revision=revision)

    print("Quantizing Flux Components")
    quantize_and_save(transformer, quantize, '/root/.cache/flux_transformer_quantized.pth')
    freeze(transformer)

    quantize_and_save(text_encoder_2, quantize, '/root/.cache/flux_text_encoder_2_quantized.pth')
    freeze(text_encoder_2)

    print("Loading FluxPipeline")

    pipe = FluxPipeline(
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_encoder_2=None,
        tokenizer_2=tokenizer_2,
        vae=vae,
        transformer=None,
    )
    pipe.text_encoder_2 = text_encoder_2
    pipe.transformer = transformer
    pipe.enable_model_cpu_offload()
    
    #dest="cuda" if torch.cuda.is_available() else "cpu"
    #print(f"FluxPipeline to {dest}")

    #pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    flux = pipe

    return pipe


def init_sd3():
    global sd3, needs_init_sd3

    needs_init_sd3 = False

    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers", 
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        variant="fp16", 
        use_safetensors=True,)
    
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    sd3 = pipe

    return pipe

def init_sdxl():
    global base, refiner, needs_init_sdxl

    needs_init_sdxl = False

    pipelines = []
    pipelines.append(
        StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, 
            variant="fp16", 
            use_safetensors=True, 
        )
    )
    pipelines.append(
        StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=pipelines[0].text_encoder_2,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            use_safetensors=True,
            variant="fp16", 
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
        #pipeline.unet.enable_xformers_memory_efficient_attention()
        #pipeline.enable_model_cpu_offload()
        pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

    base = pipelines[0]
    refiner = pipelines[1]
    return pipelines[0], pipelines[1]

def init_turbo():
    global turbo, needs_init_turbo

    needs_init_turbo = False
 
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    turbo = pipe

    return pipe

def run_flux(seed = 0, tune_with_func=None, save_output=True, reverse_swap_colors=False):
    global flux, needs_init_flux
    
    if needs_init_flux:
        init_flux()

    print(f"Seed: {seed}")
    with torch.no_grad():
        flux.generator = torch.manual_seed(seed)
        
        images = flux(
            prompt=prompt,
            #negative_prompt=negative_prompt,
            num_inference_steps=n_steps,
            height = height,
            width = width,
            max_sequence_length=512
            #guidance_scale=0.0
        ).images
        
        vars["last_run_results"] = {}

        for index, image in enumerate(images):
            exif_comment = json.dumps({
                "Model": "black-forest-labs/FLUX.1-dev",
                "Refiner": "None",
                "Seed": str(seed),
                "Steps": str(n_steps),
                "Prompt": prompt,
                #"Negative Prompt": negative_prompt,
                "Face Swapped": "True" if callable(tune_with_func) else "False",
            })
            
            comment_bytes = exif_comment.encode("utf-8")
            exif_dict = {"Exif": {}}
            exif_dict["Exif"][piexif.ExifIFD.UserComment] = comment_bytes
            exif_bytes = piexif.dump(exif_dict)
            
            if callable(tune_with_func):
                if not reverse_swap_colors:
                    results = tune_with_func(image)
                else:
                    print(np.array(image).shape)
                    results = tune_with_func(np.array(image)[:,:,::-1])

                for i, modified_image in enumerate(results):
                    file_name = f"/app/output/output-{seed}-{i}.png"
                    if save_output:
                        modified_image.save(file_name, exif=exif_bytes)
                    vars["last_run_results"][file_name] = modified_image
            else:
                file_name = f"/app/output/output-{seed}-0.png"
                if save_output:
                    image.save(file_name, exif=exif_bytes)
                vars["last_run_results"][file_name] = image
        
        return vars["last_run_results"]

def run_sd3(seed = 0, tune_with_func=None, save_output=True, reverse_swap_colors=False):
    global sd3, needs_init_sd3
    
    if needs_init_sd3:
        init_sd3()

    print(f"Seed: {seed}")
    with torch.no_grad():
        sd3.generator = torch.manual_seed(seed)
        
        images = sd3(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=n_steps,
            height = height,
            width = width,
            max_sequence_length=512
            #guidance_scale=0.0
        ).images
        
        vars["last_run_results"] = {}

        for index, image in enumerate(images):
            exif_comment = json.dumps({
                "Model": "stabilityai/stable-diffusion-3-medium-diffusers",
                "Refiner": "None",
                "Seed": str(seed),
                "Steps": str(n_steps),
                "Prompt": prompt,
                "Negative Prompt": negative_prompt,
                "Face Swapped": "True" if callable(tune_with_func) else "False",
            })
            
            comment_bytes = exif_comment.encode("utf-8")
            exif_dict = {"Exif": {}}
            exif_dict["Exif"][piexif.ExifIFD.UserComment] = comment_bytes
            exif_bytes = piexif.dump(exif_dict)
            
            if callable(tune_with_func):
                if not reverse_swap_colors:
                    results = tune_with_func(image)
                else:
                    print(np.array(image).shape)
                    results = tune_with_func(np.array(image)[:,:,::-1])

                for i, modified_image in enumerate(results):
                    file_name = f"/app/output/output-{seed}-{i}.png"
                    if save_output:
                        modified_image.save(file_name, exif=exif_bytes)
                    vars["last_run_results"][file_name] = modified_image
            else:
                file_name = f"/app/output/output-{seed}-0.png"
                if save_output:
                    image.save(file_name, exif=exif_bytes)
                vars["last_run_results"][file_name] = image
        
        return vars["last_run_results"]
    
def run_turbo(seed = 0, tune_with_func=None, save_output=True, reverse_swap_colors=False):
    global turbo, needs_init_turbo
    
    if needs_init_turbo:
        init_turbo()

    print(f"Seed: {seed}")
    with torch.no_grad():
        turbo.generator = torch.manual_seed(seed)
        
        images = turbo(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=1,
            height = 512,
            width = 512,
            guidance_scale=0.0
        ).images
        
        vars["last_run_results"] = {}

        for index, image in enumerate(images):
            exif_comment = json.dumps({
                "Model": "stabilityai/sdxl-turbo",
                "Refiner": "None",
                "Seed": str(seed),
                "Steps": str(1),
                "Prompt": prompt,
                "Negative Prompt": negative_prompt,
                "Face Swapped": "True" if callable(tune_with_func) else "False",
            })
            
            comment_bytes = exif_comment.encode("utf-8")
            exif_dict = {"Exif": {}}
            exif_dict["Exif"][piexif.ExifIFD.UserComment] = comment_bytes
            exif_bytes = piexif.dump(exif_dict)
            
            if callable(tune_with_func):
                if not reverse_swap_colors:
                    results = tune_with_func(image)
                else:
                    print(np.array(image).shape)
                    results = tune_with_func(np.array(image)[:,:,::-1])

                for i, modified_image in enumerate(results):
                    file_name = f"/app/output/output-{seed}-{i}.png"
                    if save_output:
                        modified_image.save(file_name, exif=exif_bytes)
                    vars["last_run_results"][file_name] = modified_image
            else:
                file_name = f"/app/output/output-{seed}-0.png"
                if save_output:
                    image.save(file_name, exif=exif_bytes)
                vars["last_run_results"][file_name] = image
        
        return vars["last_run_results"]


def run_sdxl(seed = 0, use_refiner=False, tune_with_func=None):
    global base, refiner, needs_init_sdxl
    
    if needs_init_sdxl:
        init_sdxl()
    
    print(f"Seed: {seed}")
    with torch.no_grad():
        base.generator = torch.manual_seed(seed)
        refiner.generator = base.generator
        if not use_refiner:
            images = base(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=n_steps,
                height = height,
                width = width,
            ).images
        else:
            images = base(
                prompt = prompt,
                negative_prompt = negative_prompt,
                num_inference_steps = n_steps,
                denoising_end =  high_noise_frac,
                output_type = "latent",
                height = height,
                width = width,
            ).images
            images = refiner(
                prompt = prompt,
                negative_prompt = negative_prompt,
                num_inference_steps = n_steps,
                denoising_start = high_noise_frac,
                image = images,
            ).images

        vars["last_run_results"] = {}

        for index, image in enumerate(images):
            exif_comment = json.dumps({
                "Model": "stabilityai/stable-diffusion-xl-base-1.0",
                "Refiner": "stabilityai/stable-diffusion-xl-refiner-1.0" if use_refiner else "None",
                "Seed": str(seed),
                "Steps": str(n_steps),
                "Prompt": prompt,
                "Negative Prompt": negative_prompt,
                "Face Swapped": "True" if callable(tune_with_func) else "False",
            })
            
            comment_bytes = exif_comment.encode("utf-8")
            exif_dict = {"Exif": {}}
            exif_dict["Exif"][piexif.ExifIFD.UserComment] = comment_bytes
            exif_bytes = piexif.dump(exif_dict)
            
            if callable(tune_with_func):
                for i, modified_image in enumerate(tune_with_func(image)):
                    file_name = f"/app/output/output-{seed}-{i}.png"
                    modified_image.save(file_name, exif=exif_bytes)
                    vars["last_run_results"][file_name] = modified_image
                    if use_refiner:
                        modified_image = modified_image.convert("RGB")
                        #Low `strength` with high `num_inference_steps` allows for cleaning image w/o losing face
                        modified_image = refiner(
                            prompt = prompt, 
                            negative_prompt = negative_prompt,
                            num_inference_steps = 1000,
                            image = modified_image, 
                            strength=refiner_strength, 
                            guidance_scale=3.5).images[0]

                        exif_comment = json.dumps({
                            "Model": "stabilityai/stable-diffusion-xl-base-1.0",
                            "Refiner": "stabilityai/stable-diffusion-xl-refiner-1.0",
                            "Refiner Strength": str(refiner_strength),
                            "Seed": str(seed),
                            "Steps": str(n_steps),
                            "Prompt": prompt,
                            "Negative Prompt": negative_prompt,
                            "Face Swapped": "True" ,
                        })
                        
                        comment_bytes = exif_comment.encode("utf-8")
                        exif_dict = {"Exif": {}}
                        exif_dict["Exif"][piexif.ExifIFD.UserComment] = comment_bytes
                        exif_bytes = piexif.dump(exif_dict)
                        file_name = f"/app/output/output-{seed}-{i}-refined.png"
                        modified_image.save(file_name, exif=exif_bytes)
                        print(modified_image)
                        vars["last_run_results"][file_name] = modified_image
            else:
                file_name = f"/app/output/output-{seed}-0.png"
                image.save(file_name, exif=exif_bytes)
                vars["last_run_results"][file_name] = image
        
        return vars["last_run_results"]

def gif_run(seed = 0, use_refiner=False, tune_with_func=None):
    global base, refiner, needs_init_sdxl
    
    if needs_init_sdxl:
        init_sdxl()
        
    images = []
    images.append(run_sdxl(seed, use_refiner, tune_with_func)[0])
    with torch.no_grad():
        refiner.generator = base.generator
        for i in range(0, 2):
            images.append(
                refiner(
                    prompt = prompt,
                    negative_prompt = negative_prompt,
                    num_inference_steps = 1000,
                    strength=0.001 + (0.001*i), 
                    guidance_scale=3.5,
                    image = images[0]
                ).images[0]
            )

    gif_bytes = export_as_gif(images, 6)
    output_path = f"/app/output/output-{seed}.gif"
    with open(output_path, "wb") as f:
        f.write(gif_bytes.getvalue())

def set_size(h=1024, w=1024):
    global height, width
    height = h // 8 * 8  # Round down to the nearest multiple of 8
    width = w // 8 * 8   # Round down to the nearest multiple of 8

def set_steps(n=30):
    global n_steps
    n_steps = n

init()

vars = {
    "run": run_sdxl, 
    "run_flux": run_flux, 
    "run_sd3": run_sd3, 
    "run_sdxl": run_sdxl, 
    "run_turbo": run_turbo, 
    "gif_run": gif_run, 
    "set_size": set_size, 
    "pipeline": base,
    "pipe": base,
    "refiner": refiner,
    "set_steps": set_steps,
    "num_images_per_prompt": num_images_per_prompt,
    "random_seed": lambda loop=1: (random.randint(0, sys.maxsize) for _ in range(loop)),
    "swap_face": lambda i: process_image(i),
    "swap_face2": lambda i, upscale=True: process_image_with_ghost(i, upscale=upscale),
    "face_permutations": face_permutations,
    "get_prompt": lambda: globals()["prompt"], 
    "set_prompt": lambda p: globals().update(prompt=p), 
    "get_negative_prompt": lambda: globals()["negative_prompt"], 
    "set_negative_prompt": lambda p: globals().update(negative_prompt=p), 
    "get_refiner_strength": lambda: globals()["refiner_strength"],
    "set_refiner_strength": lambda s: globals().update(refiner_strength=s),
    "swap_all_in_one": swap_all_in_one,
}

set_global_vars(vars)

interactive_console = code.InteractiveConsole(vars)
interactive_console.interact("Custom interactive Python session. Type 'exit' to quit.\n\n******************\nTry:\n\n[run_turbo(i) for i in random_seed(loop=10)]\n\nor\n\n[run(i) for i in random_seed(loop=3)]\n\nor\n\n[run(i, tune_with_func=swap_face) for i in random_seed(loop=3)]\n\nor\n\nx = face_permutations(3)\n\n******************\n\n")

