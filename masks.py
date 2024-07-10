import numpy as np
import cv2
from typing import Callable, List
from util import get_face_analyser
#from PIL import Image

def create_mask_from_landmarks_with_blur(image, landmarks1, landmarks2, blur_radius=20, expansion_pixels=10):
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    combined_landmarks = np.concatenate((landmarks1, landmarks2))
    landmarks_np = np.array(landmarks, dtype=np.int32)
    combined_landmarks_np = np.array(combined_landmarks, dtype=np.int32)
    hull = cv2.convexHull(combined_landmarks_np)
    cv2.fillPoly(mask, [hull], (255, 255, 255))
    kernel = np.ones((expansion_pixels, expansion_pixels), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    blurred_mask = cv2.GaussianBlur(dilated_mask, (blur_radius, blur_radius), 0)
    return blurred_mask

def get_final_image(final_frames: List[np.ndarray],
                    full_frame: np.ndarray,
                    tfm_arrays: List[np.ndarray]) -> None:
    """
    Create final video from frames
    """
    final = full_frame.copy()
        
    face_analyser = get_face_analyser()
    
    landmarks_tgt = face_analyser.get(full_frame)[0].landmark_2d_106

    for i in range(len(final_frames)):
        frame = final_frames[i][0]
        
        mat_rev = cv2.invertAffineTransform(tfm_arrays[i][0])
        swap_t = cv2.warpAffine(frame, mat_rev, (full_frame.shape[1], full_frame.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        try:
            landmarks = face_analyser.get(swap_t)[0].landmark_2d_106
            mask_t = create_mask_from_landmarks_with_blur(full_frame, landmarks, landmarks_tgt)
        except Exception as e:
            #print(f'No face in final_frames[{i}]')
            continue
        #Image.fromarray(cv2.cvtColor(swap_t,cv2.COLOR_RGBA2BGR)).save('/app/faces/tmp/swap_t.jpg')
        
        mask_t = np.expand_dims(mask_t, 2)
        
        #mask_rgb = np.repeat(mask_t, 3, axis=2)
        #if mask_rgb.max() <= 1:
        #    mask_rgb = (mask_rgb * 255).astype(np.uint8)
        #else:
        #    mask_rgb = mask_rgb.astype(np.uint8)
        #mask_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
        #Image.fromarray(cv2.cvtColor(mask_bgr,cv2.COLOR_RGBA2BGR)).save('/app/faces/tmp/mask_bgr.jpg')

        mask_normalized = mask_t.astype(np.float32) / 255
        swap_t_float = swap_t.astype(np.float32)
        final_float = final.astype(np.float32)
        blended = mask_normalized * swap_t_float + (1 - mask_normalized) * final_float
        blended_uint8 = np.clip(blended, 0, 255).astype(np.uint8)
        final = blended_uint8

    final = np.array(final, dtype='uint8')
    return final

def merge_original(img, bgr_fake, aimg, M):
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
    return fake_merged

def merge_original_with_mask(img, bgr_fake, aimg, M, external_mask=None, blend_radius=5, mask_border_thickness = 1):
    # Copy the original image to preserve the original data
    target_img = img
    
    # Assume all preprocessing steps remain the same.
    # Compute the difference between the fake background and another image, converting both to float32 before subtraction
    fake_diff = bgr_fake.astype(np.float32) - aimg.astype(np.float32)
    # Calculate the absolute difference and take the mean across the color channels, reducing it to a single-channel grayscale image
    fake_diff = np.abs(fake_diff).mean(axis=2)
    # Erase any detected difference at the top edge of the image
    fake_diff[:2,:] = 0
    # Erase any detected difference at the bottom edge of the image
    fake_diff[-2:,:] = 0
    # Remove differences at the left edge of the difference image
    fake_diff[:,:2] = 0
    # Remove differences at the right edge of the difference image
    fake_diff[:,-2:] = 0
    # Compute the inverse of the affine transformation matrix for later image alignment
    IM = cv2.invertAffineTransform(M)
    # Apply the inverse affine transformation to align the fake background image with the target image
    bgr_fake = cv2.warpAffine(bgr_fake, IM, (target_img.shape[1], target_img.shape[0]), borderMode=cv2.BORDER_REFLECT)

    if external_mask is None:
        # Generate an internal mask if no external mask is provided
        fake_diff = bgr_fake.astype(np.float32) - aimg.astype(np.float32)
        fake_diff = np.abs(fake_diff).mean(axis=2)
        fake_diff[:2,:], fake_diff[-2:,:], fake_diff[:,:2], fake_diff[:,-2:] = 0, 0, 0, 0  # Clear edges
        fake_diff = cv2.warpAffine(fake_diff, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
        img_white = np.where(fake_diff > 10, 255, 0).astype(np.float32)  # Apply threshold and ensure float32 type
        img_mask = img_white / 255  # Normalize
    else:
        # Force outer N pixels of the mask to be 0.0
        external_mask[:mask_border_thickness, :] = 0  # Top border
        external_mask[-mask_border_thickness:, :] = 0  # Bottom border
        external_mask[:, :mask_border_thickness] = 0  # Left border
        external_mask[:, -mask_border_thickness:] = 0  # Right border

        # Transform the external mask similarly to bgr_fake
        external_mask = external_mask.astype(np.float32)  # Ensure it's float32 for consistency in transformations
        external_mask = cv2.warpAffine(external_mask, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
        img_mask = np.clip(external_mask, 0, 255) / 255  # Normalize to be safe

    # Apply Gaussian blur to the mask to create a smooth transition effect
    # The 'blend_radius' determines the size of the area over which the images will be blended
    if blend_radius > 0:
        img_mask = cv2.GaussianBlur(img_mask, (blend_radius * 2 + 1, blend_radius * 2 + 1), 0)
    
    # Ensure values are within [0, 1]
    img_mask = np.clip(img_mask, 0, 1)

    # Reshape the mask for the blending operation if necessary
    if img_mask.ndim == 2:  # If the mask is 2D, add an extra dimension
        img_mask = img_mask[:, :, np.newaxis]

    # Blending the original and fake images based on the mask
    fake_merged = img_mask * bgr_fake + (1 - img_mask) * target_img.astype(np.float32)
    
    # Convert the merged image to 8-bit unsigned integers for compatibility
    fake_merged = fake_merged.astype(np.uint8)
    
    # Return the final merged image from the function
    return fake_merged

