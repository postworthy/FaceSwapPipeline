
import os
import sys
import cv2
import torch
from util import process_image_alt, process_image_alt2, preprocess_image, process_image, get_global_vars, read_and_scale, scale_image, get_face_analyser, get_face_swapper_onnx_model_and_session, create_new_directory
import random
from PIL import Image
import copy
import numpy as np
import onnx
import onnxruntime
from onnx import numpy_helper
import numpy as np
from insightface.utils import face_align
from upsampler import upsample_batch, upsample
from masks import get_final_image, merge_original
import glob
import uuid

def count_files_in_folder(directory):
    try:
        entries = os.listdir(directory)
        return sum(1 for entry in entries if os.path.isfile(os.path.join(directory, entry)))
    except Exception as e:
        return 0

#def faces_from_dir():
#    faces_directory='/app/faces/orig'
#    files = os.listdir(faces_directory)
#    image_files = [os.path.join(faces_directory, file) for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
#    return [preprocess_image(Image.fromarray(x for _, x in read_and_scale(f))) for f in image_files]

def faces_from_image_gen(names = ["young barbara palvin", "young Emma Watson", "young heidi klum", "a beautiful young teen girl"], WxH=128, tune_with_func=process_image):
    vars = get_global_vars()
    set_prompt = vars["set_prompt"]
    set_negative_prompt = vars["set_negative_prompt"]
    random_seed = vars["random_seed"]
    run_turbo = vars["run_turbo"]
    images = []

    for name in names:
        set_prompt(f'a masterpiece, intricate, highly detailed, a vibrant colorful hyper-realistic photo of ({name}), standing in a field of short grass, side swept hair, hair bun, fog and mist fill the background, a slight smirk, looking into camera')
        set_negative_prompt('long neck, ugly, distortions')
        
        results = run_turbo(next(random_seed(loop=1)), tune_with_func=tune_with_func)

        for _, (_, image) in enumerate(results.items()):
            _, img = scale_image(np.array(image), WxH)
            images.append(preprocess_image(Image.fromarray(img)))

    return images

def face_permutations(loops = 10, randomize = True, face_seed_function=faces_from_image_gen):
    source_images = face_seed_function()
    
    results = source_images[:]
    all = source_images[:]
    
    output_path = create_new_directory('/app/faces/new/')

    for i in range(0, loops):
        print(len(results))

        if i > 0 and randomize and len(results) > 10:
            print("Random Sample...")
            results = random.sample(results, 10)

        results = [x for x in get_face_permutations(results, output_path)]
        
        all.append(results)

    return all

def get_face_permutations(source_images, output_path='/app/faces/new/'):
    with torch.no_grad():

        out_count = count_files_in_folder(output_path)

        processed_pairs = set()
        
        remaining = len(source_images)
        
        while remaining > 0:
            selected_image = random.choice(source_images)
            remaining = remaining-1

            for other_image in source_images:
                if selected_image["data"] == other_image["data"]:
                    continue
                
                pair_id = (id(selected_image), id(other_image))

                if pair_id not in processed_pairs:
                    img = process_image_alt2(other_image, selected_image, True)
                    img.save(f"{output_path}/average_face_{out_count+1}.png")
                    out_count = count_files_in_folder(output_path)
                    try:
                        _, np_img = scale_image(np.array(img))
                        img = Image.fromarray(np_img)
                        yield preprocess_image(img)
                    except:
                        print(f"error preprocess_image: {output_path}/average_face_{out_count+1}.png")

                    img = process_image_alt2(selected_image, other_image, True)
                    img.save(f"{output_path}/average_face_{out_count+1}.png")
                    out_count = count_files_in_folder(output_path)
                    try:
                        _, np_img = scale_image(np.array(img))
                        img = Image.fromarray(np_img)
                        yield preprocess_image(img)
                    except:
                        print(f"error preprocess_image: {output_path}/average_face_{out_count+1}.png")

                    processed_pairs.add(pair_id)

#
# The function below reppresents the initial work to simplify and understand the 
# onnxruntime inference process for inswapper_128.onnx the hope is to find where 
# things could be better optimized. And where better resolution could be acheived.
#
# Below is a link that I used to figure out what I needed to do.
#                    
# https://github.com/deepinsight/insightface/blob/c2db41402c627cab8ea32d55da591940f2258276/python-package/insightface/model_zoo/inswapper.py#L46C1-L46C1
#
#

def save_img_fromarray(np_array, path):
    Image.fromarray(cv2.cvtColor(np_array,cv2.COLOR_RGBA2BGR)).save(path)

def profile_lambda(lambda_function):
    import time
    start_time = time.time()  # Start time before executing the lambda
    lambda_function()         # Execute the lambda function
    end_time = time.time()    # End time after execution
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")


#example_lambda = lambda: swap_all_in_one()
#example_lambda = lambda: swap_all_in_one(upsample_level = 1)
#example_lambda = lambda: swap_all_in_one(upsample_level = 2)
#example_lambda = lambda: swap_all_in_one(upsample_level = 3)
#example_lambda = lambda: swap_all_in_one(upsample_level = 4)
#profile_lambda(example_lambda)

#for i in range(0,5):
#    example_lambda = lambda: swap_all_in_one(upsample_level = i)
#    profile_lambda(example_lambda)
def single_swap(into_image, into_face, from_face, upsample_level:int=0):
    with torch.no_grad():
        face_analyser = get_face_analyser()
        model, session = get_face_swapper_onnx_model_and_session()
        emap = numpy_helper.to_array(model.graph.initializer[-1])

        input_size = tuple(session.get_inputs()[0].shape[2:4][::-1])
        input_names = [x.name for x in session.get_inputs()]
        output_names = [x.name for x in session.get_outputs()]

        final_image = None

        latent = from_face.normed_embedding.reshape((1,-1))
        latent = np.dot(latent, emap)
        latent /= np.linalg.norm(latent)

        warped_img, estimated_norm = face_align.norm_crop2(into_image, into_face.kps, input_size[0])
        
        blob = cv2.dnn.blobFromImage(warped_img, 1.0 / 255.0, input_size, (0.0, 0.0, 0.0), swapRB=True)		

        pred = session.run(output_names, {input_names[0]: blob, input_names[1]: latent})[0]
        img_fake = pred.transpose((0,2,3,1))[0]
        bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:,:,::-1]
        
        #into_image = get_final_image([[bgr_fake]], into_image, [[estimated_norm]])
        into_image = merge_original(into_image, bgr_fake, warped_img, estimated_norm)

        final_image = into_image

        if upsample_level > 0:
            if upsample_level == 1:
                final_image = upsample(final_image, True)
            elif upsample_level == 2:
                final_image = upsample(final_image, False)
            elif upsample_level == 3:
                final_image = upsample(final_image, False)
                final_image = upsample(final_image, True)
            else:
                final_image = upsample(final_image, False)
                final_image = upsample(final_image, True)
                final_image = upsample(final_image, True)
        
        #clean up cache
        torch.cuda.empty_cache()

        return final_image

def perform_swap(from_images:list, into_images:list, output_directory='/app/faces/tmp', upsample_level:int=0):
    with torch.no_grad():
        face_analyser = get_face_analyser()

        model, session = get_face_swapper_onnx_model_and_session()

        #for i, x in enumerate(model.graph.initializer):
        #    print(f'{i}: {numpy_helper.to_array(x).shape}')

        emap = numpy_helper.to_array(model.graph.initializer[-1])
        #emap = numpy_helper.to_array(model.graph.initializer[-19])

        #print(emap)
        
        input_size = tuple(session.get_inputs()[0].shape[2:4][::-1])
        input_names = [x.name for x in session.get_inputs()]
        output_names = [x.name for x in session.get_outputs()]

        #print(input_names)
        #print(output_names)

        targets = []
        not_targets = []

        for i, t in enumerate(from_images):
            sf, img = read_and_scale(t)
            faces = face_analyser.get(img)
            faces_and_latents = []
            for j, face in enumerate(faces):
                latent = face.normed_embedding.reshape((1,-1))
                latent = np.dot(latent, emap)
                latent /= np.linalg.norm(latent)
                faces_and_latents.append((face,latent))

            targets.append((sf, img, faces_and_latents))


        for i, t in enumerate(into_images):
            sf, img = read_and_scale(t)
            faces = face_analyser.get(img)
            faces_norms_blobs = []
            for j, face in enumerate(faces):
                warped_img, estimated_norm = face_align.norm_crop2(img, face.kps, input_size[0])
                blob = cv2.dnn.blobFromImage(warped_img, 1.0 / 255.0, input_size, (0.0, 0.0, 0.0), swapRB=True)		
                faces_norms_blobs.append((face, blob, estimated_norm))
                
            not_targets.append((sf, img, faces_norms_blobs))

        final_images = []

        for i, t in enumerate(targets):
            sf1, target_img, target_faces_and_latents = t
            for j, target in enumerate(target_faces_and_latents):
                target_face, latent = target
                for k, f in enumerate(not_targets):
                    sf2, img, faces_norms_blobs = f
                    for l, f_n_b in enumerate(faces_norms_blobs):
                        face, blob, estimated_norm = f_n_b
                        pred = session.run(output_names, {input_names[0]: blob, input_names[1]: latent})[0]
                        img_fake = pred.transpose((0,2,3,1))[0]
                        bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:,:,::-1]
                        
                        #merged = get_final_image([[bgr_fake]], img, [[estimated_norm]])
                        merged = merge_original(img, bgr_fake, warped_img, estimated_norm)
                        img = merged
                    
                    final_images.append(img)
        
        if upsample_level > 0:
            if upsample_level == 1:
                final_images = upsample_batch(final_images, 2)
                [save_img_fromarray(f, f'{output_directory}/{i}x2.jpg') for i, f in enumerate(final_images)]
            elif upsample_level == 2:
                final_images = upsample_batch(final_images, 4)
                [save_img_fromarray(f, f'{output_directory}/{i}x4.jpg') for i,f in enumerate(final_images)]
            elif upsample_level == 3:
                final_images = upsample_batch(final_images, 4)
                final_images = upsample_batch(final_images, 2)
                [save_img_fromarray(f, f'{output_directory}/{i}x8.jpg') for i,f in enumerate(final_images)]
            else:
                final_images = upsample_batch(final_images, 4)
                final_images = upsample_batch(final_images, 2)
                final_images = upsample_batch(final_images, 2)
                [save_img_fromarray(f, f'{output_directory}/{i}x16.jpg') for i,f in enumerate(final_images)]
        else:
            [save_img_fromarray(f, f'{output_directory}/{i}.jpg') for i,f in enumerate(final_images)]

        #clean up cache
        torch.cuda.empty_cache()

        return final_images


def swap_all_in_one(faces_directory='/app/faces/', upsample_level:int=0):
    from_images = [os.path.join(faces_directory, file) for file in os.listdir(faces_directory) if '_target_' in file and file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    into_images = [os.path.join(faces_directory, file) for file in os.listdir(faces_directory) if '_target_' not in file and file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    return perform_swap(from_images, into_images, upsample_level=upsample_level)

def _OLD_swap_all_in_one_OLD_(faces_directory='/app/faces/', save_intermediates=False, upsample_level:int=0):
    import onnx
    import onnxruntime
    from onnx import numpy_helper
    import numpy as np
    from util import get_face_analyser
    from insightface.utils import face_align
    from upsampler import upsample

    providers = onnxruntime.get_available_providers()

    if 'TensorrtExecutionProvider' in providers:
        providers.remove('TensorrtExecutionProvider')

    face_analyser = get_face_analyser()

    model_file = '/root/.insightface/models/inswapper_128.onnx'
    model = onnx.load(model_file)
    session = onnxruntime.InferenceSession(model_file, None)

    graph = model.graph
    emap = numpy_helper.to_array(graph.initializer[-1])
    input_size = tuple(session.get_inputs()[0].shape[2:4][::-1])
    input_names = [x.name for x in session.get_inputs()]
    output_names = [x.name for x in session.get_outputs()]

    targets = []
    not_targets = []

    for i, t in enumerate([os.path.join(faces_directory, file) for file in os.listdir(faces_directory) if '_target_' in file and file.lower().endswith(('.png', '.jpg', '.jpeg'))]):
        sf, img = read_and_scale(t)
        if save_intermediates and sf < 1.0 :
            save_img_fromarray(img, f'/app/faces/tmp/test_scaled_target_{i}.jpg') 
        faces = face_analyser.get(img)
        faces_and_latents = []
        for j, face in enumerate(faces):
            latent = face.normed_embedding.reshape((1,-1))
            latent = np.dot(latent, emap)
            latent /= np.linalg.norm(latent)
            faces_and_latents.append((face,latent))

        targets.append((sf, img, faces_and_latents))


    for i, t in enumerate([os.path.join(faces_directory, file) for file in os.listdir(faces_directory) if '_target_' not in file and file.lower().endswith(('.png', '.jpg', '.jpeg'))]):
        sf, img = read_and_scale(t)
        if save_intermediates and sf < 1.0:
            save_img_fromarray(img, f'/app/faces/tmp/test_scaled_not_target_{i}.jpg') 
        faces = face_analyser.get(img)
        faces_norms_blobs = []
        for j, face in enumerate(faces):
            warped_img, estimated_norm = face_align.norm_crop2(img, face.kps, input_size[0])
            blob = cv2.dnn.blobFromImage(warped_img, 1.0 / 255.0, input_size, (0.0, 0.0, 0.0), swapRB=True)		
            faces_norms_blobs.append((face, blob, estimated_norm))
            if save_intermediates:
                if upsample_level > 0:
                    upsampled_warped = upsample(warped_img, False if upsample_level > 1 else True)
                    save_img_fromarray(upsampled_warped, f'/app/faces/tmp/test_not_target__{i}_{j}.jpg')
                else:
                    save_img_fromarray(warped_img, f'/app/faces/tmp/test_not_target__{i}_{j}.jpg')
            
        not_targets.append((sf, img, faces_norms_blobs))

    for i, t in enumerate(targets):
        sf1, target_img, target_faces_and_latents = t
        for j, target in enumerate(target_faces_and_latents):
            target_face, latent = target
            for k, f in enumerate(not_targets):
                sf2, img, faces_norms_blobs = f
                for l, f_n_b in enumerate(faces_norms_blobs):
                    face, blob, estimated_norm = f_n_b
                    pred = session.run(output_names, {input_names[0]: blob, input_names[1]: latent})[0]
                    img_fake = pred.transpose((0,2,3,1))[0]
                    bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:,:,::-1]

                    if save_intermediates:
                        if upsample_level > 0:
                            upsampled_final = upsample(bgr_fake, False if upsample_level > 1 else True)
                            save_img_fromarray(upsampled_final, f'/app/faces/tmp/test_swap_{i}_{j}_{k}_{l}.jpg')
                        else:
                            save_img_fromarray(bgr_fake, f'/app/faces/tmp/test_swap_{i}_{j}_{k}_{l}.jpg')

                    merged = merge_original(img, bgr_fake, warped_img, estimated_norm)
                    if upsample_level > 0:
                        if upsample_level == 1:
                            merged = upsample(merged, True)
                            save_img_fromarray(merged, f'/app/faces/tmp/test_mergedx2_{i}_{j}_{k}_{l}.jpg')
                        elif upsample_level == 2:
                            merged = upsample(merged, False)
                            save_img_fromarray(merged, f'/app/faces/tmp/test_mergedx4_{i}_{j}_{k}_{l}.jpg')
                        elif upsample_level == 3:
                            merged = upsample(merged, False)
                            merged = upsample(merged, True)
                            save_img_fromarray(merged, f'/app/faces/tmp/test_mergedx8_{i}_{j}_{k}_{l}.jpg')
                        else:
                            merged = upsample(merged, False)
                            merged = upsample(merged, True)
                            merged = upsample(merged, True)
                            save_img_fromarray(merged, f'/app/faces/tmp/test_mergedx16_{i}_{j}_{k}_{l}.jpg')
                    else:
                        save_img_fromarray(merged, f'/app/faces/tmp/test_merged_{i}_{j}_{k}_{l}.jpg')
                    
def generate_image_prompt():
    # Boolean variables with random values
    is_male = random.choice([True, False])
    is_attractive = random.choice([True, False])
    # Arrays of strings with default values
    ethnicity = [
    "Caucasian", "Hispanic", "Black", "Middle-Eastern", "South Asian", 
    "East Asian","Southeast Asian","Pacific Islander","Native American","Indigenous Australian",
    "African","Caribbean","Central Asian","Slavic","Mediterranean",
    "Scandinavian","Baltic","Latin American","Mixed","Berber",
    "Turkish","Korean","Japanese","Mongolian","Polynesian","Micronesian",
    "Melanesian","Indigenous Arctic (e.g., Inuit)","Sami","Tibetan","Filipino",
    "Indonesian","Malaysian","Siberian","West African","East African",
    "North African","Southern African","Central African","Brazilian","Argentinian",
    "Chilean","Peruvian","Colombian","Venezuelan","Ecuadorian",
    "Uruguayan","Paraguayan","Bolivian","Greek","Portuguese",
    "Spanish","Italian","French","German","Dutch",
    "Belgian","Austrian","Swiss","British","Irish",
    "Icelandic","Norwegian","Swedish","Finnish","Danish",
    "Russian","Ukrainian","Belarusian","Polish","Czech",
    "Slovak","Hungarian","Romanian","Bulgarian","Albanian",
    "Croatian","Serbian","Bosnian","Montenegrin","Macedonian",
    "Slovenian","Estonian","Latvian","Lithuanian","Georgian",
    "Armenian","Azerbaijani","Kazakh","Uzbek","Turkmen",
    "Kyrgyz","Tajik","Iranian","Iraqi","Syrian",
    "Lebanese","Jordanian","Palestinian","Israeli","Saudi Arabian",
    "Yemeni","Omani","Emirati","Qatari","Bahraini",
    "Kuwaiti","Egyptian","Libyan","Tunisian","Algerian",
    "Moroccan","Sudanese","Ethiopian","Somali","Kenyan",
    "Ugandan","Tanzanian","Rwandan","Burundian","Malawian",
    "Zambian","Zimbabwean","Botswanan","Namibian","South African",
    "Lesotho","Eswatini","Mozambican","Angolan","Congolese",
    "Gabonese","Cameroonian","Nigerian","Ghanaian","Ivorian",
    "Malian","Burkinabe","Senegalese","Guinean","Sierra Leonean",
    "Liberian","Indian","Pakistani","Bangladeshi","Sri Lankan",
    "Nepalese","Bhutanese","Maldivian","Thai","Vietnamese",
    "Laotian","Cambodian","Myanmar (Burmese)","Singaporean","Bruneian",
    "Timorese","Chinese","Taiwanese","Hong Kong","Macanese"
    ]
    hair_style = [
    "short", "long", "curly", "straight", "wavy", "bald", "ponytail",
    "buzz cut", "bob", "pixie cut", "afro", "braids", "cornrows",
    "dreadlocks", "undercut", "side part", "middle part", "crew cut",
    "mohawk", "frohawk", "bun", "top knot", "shaved sides",
    "layered", "shaggy", "slicked back", "spiky", "quiff",
    "pompadour", "fishtail braid", "French braid", "box braids",
    "Senegalese twist", "faux hawk", "mullet", "taper fade",
    "high and tight", "man bun", "lob (long bob)", "V-cut",
    "U-cut", "emoboy", "scene hair", "flapper bob", "finger waves",
    "beehive", "bowl cut", "pageboy", "pixie bob", "feathered hair",
    "half updo", "half down", "double buns", "space buns",
    "twists", "rattail", "perm", "jheri curl", "blowout",
    "comb over", "slick back", "faux locs", "yarn braids", "bantu knots"
    ]
    face_shape = [
    "oval", "square", "round", "diamond", "heart", "rectangular",
    "triangular", "oblong", "pear-shaped", "teardrop",
    "inverted triangle", "soft rectangle", "rounded square",
    "elongated oval", "soft heart", "sharp heart", "wide square",
    "narrow oval", "chiseled", "angular", "rounded",
    "softly curved", "wide round", "slim diamond", "broad diamond",
    "gentle oval", "prominent jawline", "soft jawline",
    "high cheekbones", "low cheekbones", "flat cheekbones",
    "rounded cheeks", "slim cheeks", "full cheeks",
    "pointed chin", "rounded chin", "square chin",
    "slender face", "broad face", "tapered face",
    "full face", "narrow face", "delicate features",
    "strong features", "pronounced chin", "subtle chin"
    ]
    face_expression = [
    "happy", "sad", "angry", "surprised", "neutral", "confused",
    "smiling", "frowning", "laughing", "crying", "skeptical",
    "bored", "amused", "anxious", "excited", "fearful",
    "disgusted", "contemptuous", "flirty", "pensive", "determined",
    "exhausted", "relieved", "hopeful", "dreamy", "proud",
    "ashamed", "guilty", "envious", "sympathetic", "curious",
    "alarmed", "bewildered", "charmed", "apathetic", "indifferent",
    "sulking", "impressed", "startled", "melancholic", "nostalgic",
    "wistful", "horrified", "ecstatic", "despondent", "irritated",
    "embarrassed", "shocked", "woeful", "blissful", "content"
    ]
    looking_in_direction = ["left", "right", "straight", "up", "down"]
    age_in_years = [str(age) for age in range(10, 40)]  # Ages 18 to 70
    location = [
    "standing alone in a field of tall grass",
    "sitting solitary on a rocky beach",
    "leaning against an old brick wall in an abandoned alley",
    "wandering alone in a misty forest",
    "perched atop a hill overlooking a deserted valley",
    "isolated on a small boat in the middle of a calm lake",
    "on a deserted urban rooftop under the night sky",
    "at the edge of a cliff facing the endless ocean",
    "in the heart of a barren desert, with nothing but sand for miles",
    "in a quiet, snow-covered meadow under a starry night",
    "sitting alone on a bench in an empty city park at dusk",
    "standing on a long, deserted highway stretching into the horizon",
    "on a secluded balcony overlooking a sleeping city",
    "in an ancient, forgotten temple overgrown with vines",
    "at the end of a long, wooden pier gazing at a distant storm",
    "alone in a vibrant flower garden, surrounded by butterflies",
    "on a high mountain peak, clouds rolling below",
    "in the ruins of an old castle engulfed by a thick fog",
    "sitting at the edge of a tranquil pond, reflecting the clear blue sky",
    "in a vast library, filled with towering bookshelves and no one else in sight"
    ]
    # Random selection from each array
    selected_ethnicity = random.choice(ethnicity)
    selected_hair_style = random.choice(hair_style)
    selected_face_shape = random.choice(face_shape)
    selected_face_expression = random.choice(face_expression)
    selected_looking_direction = random.choice(looking_in_direction)
    selected_age = random.choice(age_in_years)
    selected_location = random.choice(location)
    # Constructing the prompt
    prompt = f"A hyper realistic photo, A {selected_ethnicity}, {selected_age}-year-old {'male' if is_male else 'female'} model with {selected_hair_style} hair, " \
             f"{selected_face_shape} face, and a {selected_face_expression} expression, looking {selected_looking_direction}, " \
             f"{'extremely ' if is_attractive else ' '}attractive, {selected_location}."
    return prompt

def gen_new_faces(loop=10):
    face_analyser = get_face_analyser()
    vars = get_global_vars()
    set_prompt = vars["set_prompt"]
    set_negative_prompt = vars["set_negative_prompt"]
    random_seed = vars["random_seed"]
    run_turbo = vars["run_turbo"]

    for i in range(0, loop):
        prompt=generate_image_prompt()
        set_prompt(prompt)
        set_negative_prompt("deformed, ugly, low resolution, bad quality, ugly face, bad teeth")
        for j in random_seed(loop=1):
            results = run_turbo(j, save_output=False)
            for file_name, image in results.items():
                image = np.array(image)
                try:
                    face = face_analyser.get(image)[0]
                    if face:
                        warped_img, _ = face_align.norm_crop2(image, face.kps, 256)
                        Image.fromarray(warped_img).save(file_name)
                except Exception as e:
                    print(e)

def convert_to_sketch(image_path, invert=True):
    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted_img = 255 - gray_img
    blurred_img = cv2.GaussianBlur(inverted_img, (21, 21), 0)
    inverted_blurred = 255 - blurred_img
    sketch_img = cv2.divide(gray_img, inverted_blurred, scale=256.0)
    if invert:
        sketch_img = 255 - sketch_img
    return cv2.cvtColor(sketch_img, cv2.COLOR_GRAY2BGR)

def gen_new_faces_from_faces():
    from diffusers import AutoPipelineForImage2Image

    pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    pipe.to("cuda")

    faces_directory='/app/faces/'
    files = os.listdir(faces_directory)
    image_files = [os.path.join(faces_directory, file) for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for i, f in enumerate(image_files):
        try:
            init_image = cv2.resize(cv2.imread(f), (512, 512))
            #init_image = cv2.cvtColor(cv2.Canny(cv2.imread(f), 100, 200), cv2.COLOR_GRAY2BGR)
            image = pipe('a beautiful woman face, 8k, raw', image=init_image, num_inference_steps=20, strength=0.75, guidance_scale=6.0).images[0]
            image.save(f'/app/output/face_from_face-{i}.jpg')
        except Exception as e:
            print(e)

def get_celebrity_prompts():
    from imdb import Cinemagoer
    ia = Cinemagoer()
    # Fetch top 250 movies as a sample
    top_movies = ia.get_top250_movies()

    # Randomly shuffle the list of top movies
    random.shuffle(top_movies)
    
    # Loop through top movies and fetch their cast
    for movie in top_movies:
        # Get movie details
        movie = ia.get_movie(movie.movieID)
        movie_title = movie.get('title', 'Unknown Title')

        # Process each cast member
        if movie.get('cast'):
            for actor in movie['cast']:
                actor_name = actor.get('name', 'Unknown Actor')
                character_name = actor.currentRole if actor.currentRole is not None else 'Unknown Character'

                # Format and add the string to the list
                role_description = f"{actor_name} in the role of {character_name} from the movie {movie_title}"
                yield role_description

def get_celebrity_faces(count=1, WxH=256):
    vars = get_global_vars()
    set_prompt = vars["set_prompt"]
    set_negative_prompt = vars["set_negative_prompt"]
    random_seed = vars["random_seed"]
    run_turbo = vars["run_turbo"]

    prompts = get_celebrity_prompts()
    for i, prompt in enumerate(prompts):
        set_prompt(prompt)
        set_negative_prompt("long neck, (worst quality, low quality:1.4), (malformed hands:1.4),(poorly drawn hands:1.4),(mutated fingers:1.4),(extra limbs:1.35),(poorly drawn face:1.4), ugly, huge eyes, fat, indoor, worst face,text,watermark,(worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, skin spots, acnes, skin blemishes, bad anatomy, DeepNegative, facing away, tilted head, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worstquality, low quality, normal quality, jpegartifacts, signature, watermark, username, blurry, bad feet, cropped, poorly drawn hands, poorly drawn face, mutation, deformed, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, extra fingers, fewer digits, extra limbs, extra arms,extra legs, malformed limbs, fused fingers, too many fingers, long neck, cross-eyed,mutated hands, polar lowres, bad body, bad proportions, gross proportions, text, error, missing fingers, missing arms, missing legs, extra digit, extra arms, extra leg, extra foot, (repeating hair)")
        for seed in random_seed(loop=3):
            results = run_turbo(seed, save_output=False)

            for _, (file_name, image) in enumerate(results.items()):
                _, img = scale_image(np.array(image), WxH)
                pre = preprocess_image(Image.fromarray(img))
                pre_image = np.array(pre["data"])
                pre_face = pre["first_face"]
                if pre_face:
                    into_img, _ = face_align.norm_crop2(pre_image, pre_face.kps, 256)
                    Image.fromarray(into_img).save(file_name)
        if i >= count:
            break

def permute_from_dir(into_image_dir='/sdxl_turbo_faces', from_image_dir='/img_align_celeba',  output_dir='/app/output/'):
    run_id = str(uuid.uuid4())
    border_size=50
    vars = get_global_vars()
    random_seed = vars["random_seed"]
    swap_face = vars["swap_face"]

    into_image_dir = glob.glob(f'{into_image_dir}/output-*.*g')
    from_images_list = glob.glob(f'{from_image_dir}/*.*g')

    for i, f in enumerate(into_image_dir):
        img = cv2.copyMakeBorder(cv2.imread(f), top=border_size, bottom=border_size, left=border_size, right=border_size, borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
        into_img = preprocess_image(img)
        for j in range(0, 1):
            try:
                from_img = preprocess_image(cv2.imread(random.choice(from_images_list)))
            except:
                continue
            if into_img['first_face'] and from_img['first_face']:
                output_img = process_image_alt(into_img, from_img)
                output_img = preprocess_image(output_img)
                if output_img['first_face']:
                    #combined = np.concatenate([output_img['data'], into_img['data']], axis=1)
                    #result = Image.fromarray(np.array(combined)[:,:,::-1])
                    result = Image.fromarray(np.array(output_img['data'])[:,:,::-1])
                    result.save(f'{output_dir}/{run_id}_{i}_{j}.jpg')
                    into_img = output_img
                else:
                    break

        