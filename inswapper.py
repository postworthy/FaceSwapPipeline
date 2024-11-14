import time
import numpy as np
import onnxruntime
import cv2
import onnx
from onnx import numpy_helper
import insightface

class INSwapper():
    def __init__(self, model_file=None, session=None, batch_size=16):
        self.model_file = model_file
        self.session = session
        self.batch_size = batch_size
        print("Batch Size:", self.batch_size)
        # Load and modify the ONNX model
        model = onnx.load(self.model_file)
        graph = model.graph

        # Set batch size to 8
        for input_tensor in graph.input:
            tensor_type = input_tensor.type.tensor_type
            shape = tensor_type.shape

            if len(shape.dim) > 0:
                # Set batch size to batch_size
                shape.dim[0].dim_value = batch_size

                # Or set batch size to dynamic
                # shape.dim[0].dim_param = 'batch_size'

        # Check the modified model
        onnx.checker.check_model(model)

        # Serialize the model and create the session
        model_bytes = model.SerializeToString()
        if self.session is None:
            # Set session options if needed
            sess_options = onnxruntime.SessionOptions()
            # Optional: Set graph optimization level
            sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

            # Specify the execution providers in order of preference
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = onnxruntime.InferenceSession(model_bytes, sess_options, providers=providers)


        # Verify that the GPU provider is being used
        print("Available providers:", onnxruntime.get_available_providers())
        print("Session providers:", self.session.get_providers())

        # Extract model parameters
        self.emap = numpy_helper.to_array(graph.initializer[-1])
        self.input_mean = 0.0
        self.input_std = 255.0

        # Get input and output names
        inputs = self.session.get_inputs()
        self.input_names = [inp.name for inp in inputs]
        outputs = self.session.get_outputs()
        self.output_names = [out.name for out in outputs]
        assert len(self.output_names) == 1

        # Get input shape
        input_cfg = inputs[0]
        self.input_shape = input_cfg.shape
        print('inswapper-shape:', self.input_shape)
        self.input_size = tuple(self.input_shape[2:4][::-1])

    def forward(self, imgs, latents):
        # Ensure imgs and latents are numpy arrays
        imgs = np.asarray(imgs)
        latents = np.asarray(latents)

        # Normalize the images
        imgs = (imgs - self.input_mean) / self.input_std

        # Run the model
        preds = self.session.run(self.output_names, {
            self.input_names[0]: imgs,
            self.input_names[1]: latents
        })[0]

        return preds

    def get(self, imgs, target_faces, source_faces, paste_back=True):
        # Process batch inputs
        blobs = []
        latents = []
        Ms = []
        aimgs = []
        is_single=False
        

        if not isinstance(imgs, list):
            imgs = [imgs] 
            is_single=True
        if not isinstance(target_faces, list):
            target_faces = [target_faces] * len(imgs)
        if not isinstance(source_faces, list):
            source_faces = [source_faces] * len(imgs)

        #print(f"imgs shape: {imgs[0].shape}")

        while len(imgs) < self.batch_size:
            imgs.append(imgs[-1])

        while len(target_faces) < len(imgs):
            target_faces.append(target_faces[-1])

        while len(source_faces) < len(imgs):
            source_faces.append(source_faces[-1])

        for idx in range(len(imgs)):
            img = imgs[idx]
            target_face = target_faces[idx]
            source_face = source_faces[idx]

            aimg, M = insightface.utils.face_align.norm_crop2(img, target_face.kps, self.input_size[0])
            
            blob = cv2.dnn.blobFromImage(
                aimg,
                1.0 / self.input_std,
                self.input_size,
                (self.input_mean, self.input_mean, self.input_mean),
                swapRB=True
            )
            latent = source_face.normed_embedding.reshape((1, -1))
            latent = np.dot(latent, self.emap)
            latent /= np.linalg.norm(latent)

            aimgs.append(aimg)
            blobs.append(blob)
            latents.append(latent)
            Ms.append(M)


        # Stack blobs and latents
        blobs = np.vstack(blobs)
        latents = np.vstack(latents)

        #print(f"Blobs shape: {blobs.shape}")
        #print(f"Latents shape: {latents.shape}")

        # Run the model
        preds = self.session.run(self.output_names, {
            self.input_names[0]: blobs,
            self.input_names[1]: latents
        })[0]

        #print(f"preds shape: {preds.shape}")

        # Postprocess the outputs
        fake_images = []
        for i in range(len(preds)):
            pred = preds[i]
            img_fake = pred.transpose((1, 2, 0))
            bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:, :, ::-1]
            if not paste_back:
                fake_images.append((bgr_fake, Ms[i]))
            else:
                img = imgs[i]
                aimg = aimgs[i]
                M = Ms[i]
                
                target_img = img
                fake_diff = bgr_fake.astype(np.float32) - aimg.astype(np.float32)
                fake_diff = np.abs(fake_diff).mean(axis=2)
                fake_diff[:2,:] = 0
                fake_diff[-2:,:] = 0
                fake_diff[:,:2] = 0
                fake_diff[:,-2:] = 0
                IM = cv2.invertAffineTransform(M)
                img_white = np.full((aimg.shape[0],aimg.shape[1]), 255, dtype=np.float32)
                bgr_fake = cv2.warpAffine(bgr_fake, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
                img_white = cv2.warpAffine(img_white, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
                fake_diff = cv2.warpAffine(fake_diff, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
                img_white[img_white>20] = 255
                fthresh = 10
                fake_diff[fake_diff<fthresh] = 0
                fake_diff[fake_diff>=fthresh] = 255
                img_mask = img_white
                mask_h_inds, mask_w_inds = np.where(img_mask==255)
                mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
                mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
                mask_size = int(np.sqrt(mask_h*mask_w))
                k = max(mask_size//10, 10)
                #k = max(mask_size//20, 6)
                #k = 6
                kernel = np.ones((k,k),np.uint8)
                img_mask = cv2.erode(img_mask,kernel,iterations = 1)
                kernel = np.ones((2,2),np.uint8)
                fake_diff = cv2.dilate(fake_diff,kernel,iterations = 1)
                k = max(mask_size//20, 5)
                #k = 3
                #k = 3
                kernel_size = (k, k)
                blur_size = tuple(2*i+1 for i in kernel_size)
                img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
                k = 5
                kernel_size = (k, k)
                blur_size = tuple(2*i+1 for i in kernel_size)
                fake_diff = cv2.GaussianBlur(fake_diff, blur_size, 0)
                img_mask /= 255
                fake_diff /= 255
                #img_mask = fake_diff
                img_mask = np.reshape(img_mask, [img_mask.shape[0],img_mask.shape[1],1])
                fake_merged = img_mask * bgr_fake + (1-img_mask) * target_img.astype(np.float32)
                fake_merged = fake_merged.astype(np.uint8)

                fake_images.append(fake_merged)
                


        if is_single:
            return fake_images[0]
        else:
            return fake_images