import tensorflow as tf
import cv2
import numpy as np

import sys
sys.path.insert(0,'./')
from utils.utils import  get_percentile_mask, fast_clip

def tf_preprocess_synth(x_LD, x_HD, x_mask, ppe):
    # Randomly generate negative image
    p = 0.0
    negative = np.random.binomial(1,p)
    
    x_LD = x_LD.numpy()
    x_mask = x_mask.numpy()
    x_HD = x_HD.numpy().astype('float32')
    if negative:
        x_LD = 255-x_LD
        x_HD = 255-x_HD

    x_LD, x_HD, x_mask, ppe = random_rotation_rigid(x_LD, x_HD, x_mask, ppe)
    blur_sigma = abs(np.random.normal(0, 0.25))+0.2
    noise_sigma = abs(np.random.normal(0, 2.5))+1
    additive_noise = np.random.normal(0,noise_sigma, x_LD.shape).astype('float32')
    x_LD = cv2.GaussianBlur(x_LD, (11,11), blur_sigma)
    x_LD = cv2.add(x_LD, additive_noise, dtype=cv2.CV_8U)
    #x_LD = apply_jpeg(x_LD, np.random.uniform(min_jpeg_quality, 100))
    p_low = get_percentile_mask(x_LD, None, 1)
    p_high = get_percentile_mask(x_LD, None, 98)
    x_LD = (x_LD.astype('float32')-p_low)/(p_high-p_low+1e-10)
    x_LD = fast_clip(x_LD,0,1)
    x_HD = (x_HD-p_low)/(p_high-p_low+1e-10)
    x_HD = fast_clip(x_HD,0,1)
    return x_LD[:, :, np.newaxis], x_HD[:, :, np.newaxis], x_mask[:, :, np.newaxis]/255.0+0.1, ppe

def random_rotation_rigid(x_LD, x_HD, x_mask, ppe):
    a = np.random.uniform(0,1)
    if a > 0.5:
        x_LD = x_LD[::-1]
        x_HD = x_HD[::-1]
        x_mask = x_mask[::-1]

    k = np.random.randint(0,4)
    x_LD = np.rot90(x_LD, k=k)
    x_HD = np.rot90(x_HD, k=k)
    x_mask = np.rot90(x_mask, k=k)
    return x_LD, x_HD, x_mask, ppe

def random_rotate_uniform(img, polygon, ppe):
    angle = np.random.uniform(0,360)
    H,W = img.shape
    M = cv2.getRotationMatrix2D((W//2, H//2), angle, 1.0)
    img_rotated = cv2.warpAffine(img.numpy(), M, (W,H), borderMode=cv2.BORDER_REPLICATE)
    P = np.zeros((3,3))
    P[:2] = M
    P[-1,-1]=1
    rotated_polygon = cv2.perspectiveTransform(polygon.numpy()[np.newaxis].astype('float32'), P).astype('float32')
    return img_rotated, rotated_polygon[0], ppe

def change_contrast(img, polygon, ppe):
    H,W = img.numpy().shape
    polygon_center = np.mean(polygon.numpy(), axis=0)
    polygon = np.float32(polygon-polygon_center+np.array([W//2, H//2]))
    
    old_min = get_percentile_mask(img.numpy(), polygon, 1)
    old_max = get_percentile_mask(img.numpy(), polygon, 98)
    
    target_min = np.random.uniform(0, 60)
    target_max = np.random.uniform(target_min+30, 255)
    img_contrast = np.subtract(img.numpy().astype('float32'),old_min)/(old_max-old_min)*(target_max-target_min)+target_min
    return img_contrast, polygon, ppe

def blur(img, polygon, ppe):
    ppe_mean = float(ppe.numpy()[0])
    #blur_sigma = np.random.uniform(0.01, 0.3)*ppe_mean
    #ker_size = int(np.ceil(blur_sigma*3))*2+1
    blur_sigma = 0
    ker_size = 3
    img_HD = img.numpy()
    img_LD = cv2.GaussianBlur(img_HD, (ker_size,ker_size), blur_sigma)
    return img_LD, img_HD, polygon, ppe

def resize(img_LD, img_HD, polygon, ppe, H_window=64, W_window=64, ppe_min=1.0, ppe_max=1.5):
    original_ppe = ppe.numpy()
    target_ppe = np.random.uniform(ppe_min, ppe_max)

    H, W = img_LD.shape
    H_new = int(np.round(H * target_ppe / original_ppe))
    W_new = int(np.round(W * target_ppe / original_ppe))
    H_new2 = H_new*2
    W_new2 = W_new*2
    H_window2 = H_window*2
    W_window2 = W_window*2
    img_LD = cv2.resize(img_LD.numpy(), (W_new, H_new), interpolation=cv2.INTER_AREA)
    img_HD = cv2.resize(img_HD.numpy(), (W_new2, H_new2), interpolation=cv2.INTER_AREA)
    polygon = (polygon.numpy() * target_ppe) / original_ppe

    #Cropping img_LD
    if H_new < H_window:
        pad1 = (H_window - H_new) // 2
        pad2 = H_window - H_new - pad1
        img_LD = np.pad(img_LD, ((pad1, pad2), (0, 0)), mode='edge')
        polygon += np.array([pad1, 0])
    elif H_new > H_window:
        pad1 = (H_new - H_window) // 2
        img_LD = img_LD[pad1:pad1 + H_window]
        polygon -= np.array([pad1, 0])
    if W_new < W_window:
        pad1 = (W_window - W_new) // 2
        pad2 = W_window - W_new - pad1
        img_LD = np.pad(img_LD, ((0, 0), (pad1, pad2)), mode='edge')
        polygon += np.array([0, pad1])
    elif H_new > H_window:
        pad1 = (W_new - W_window) // 2
        img_LD = img_LD[:, pad1:pad1 + W_window]
        polygon -= np.array([0, pad1])

    img_LD = img_LD[:H_window, :W_window]
    
    
    #Cropping img_HD 
    if H_new2 < H_window2:
        pad1 = (H_window2 - H_new2) // 2
        pad2 = H_window2 - H_new2 - pad1
        img_HD = np.pad(img_HD, ((pad1, pad2), (0, 0)), mode='edge')
    elif H_new2 > H_window2:
        pad1 = (H_new2 - H_window2) // 2
        img_HD = img_HD[pad1:pad1 + H_window2]
    if W_new2 < W_window2:
        pad1 = (W_window2 - W_new2) // 2
        pad2 = W_window2 - W_new2 - pad1
        img_HD = np.pad(img_HD, ((0, 0), (pad1, pad2)), mode='edge')
    elif H_new2 > H_window2:
        pad1 = (W_new2 - W_window2) // 2
        img_HD = img_HD[:, pad1:pad1 + W_window2]

    img_HD = img_HD[:H_window2, :W_window2]

    return img_LD, img_HD, polygon, target_ppe

 
    
def add_noise(img_LD, img_HD, polygon, ppe, apply_contrast_stretch=True, bypass_noise=False):
    img_LD = fast_clip(img_LD.numpy(),0,500)
    polygon=polygon.numpy().astype('float32')
    poisson_strength = np.random.uniform(0.03, 0.16)
    white_noise_sigma = np.random.uniform(0.5, 2.5)
    salt_prob = 2*10**np.random.uniform(-2,-5)
    salt_strength = np.random.uniform(4,11)
    mask = np.random.uniform(0,1,img_LD.shape)>(1-salt_prob)
    
    noisy_img =  np.random.poisson(img_LD / poisson_strength)*poisson_strength + np.random.normal(0, white_noise_sigma, img_LD.shape)
    sat_noise = np.random.uniform(0, salt_strength, noisy_img.shape)*mask
    noisy_img += sat_noise
    if bypass_noise:
        img_LD = fast_clip(img_LD, 0, 255).astype('uint8')
    else:
        img_LD = fast_clip(noisy_img, 0, 255).astype('uint8')
    img_HD = fast_clip(img_HD.numpy(), 0, 255).astype('uint8')
    if apply_contrast_stretch:
        p_low = get_percentile_mask(img_LD, None, 1)
        p_high = get_percentile_mask(img_LD, None, 98)
        img_LD = (img_LD.astype('float32')-p_low)/(p_high-p_low+1e-10)
        img_LD = fast_clip(img_LD,0,1)
        img_HD = (img_HD.astype('float32')-p_low)/(p_high-p_low+1e-10)
        img_HD = fast_clip(img_HD,0,1)
    else:
        img_LD /= 255.0
        img_HD /= 255.0
    
    img_mask = np.ones_like(img_HD)*1e-4
    img_mask = cv2.fillPoly(img_mask, [np.int32(polygon*2)], 1, lineType = cv2.LINE_AA)
    img_mask = cv2.GaussianBlur(img_mask, (3,3), 0)
    return img_LD[:,:,np.newaxis], img_HD[:,:,np.newaxis], img_mask[:,:,np.newaxis], ppe

def filter_blurred_images(img_LD, img_HD, img_mask, ppe):
    HD_lap = cv2.Laplacian(img_HD.numpy(), -1, ksize=3)
    img_mask = img_mask.numpy()
    if HD_lap[img_mask > 0.5].std() < 0.7: #Found experimentally
        img_mask = img_mask*0
    return img_LD, img_HD.numpy()[:,:,np.newaxis], img_mask[:,:,np.newaxis], ppe