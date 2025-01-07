import numpy as np
import cv2

def masked_mse(im1, im2, mask=None):
    mse_full = (im1.astype('float32')-im2.astype('float32'))**2
    if mask is None:
        mask = np.ones_like(im1)
    summ = np.sum(mask)
    return np.sum(mse_full*mask)/summ

def masked_psnr(im1, im2, mask=None, R=1):
    mask_mse = masked_mse(im1, im2, mask)
    return 10*np.log10(R**2/mask_mse)

def ssim(im1,im2, win_size=7, dynamic_range = 1.0):
    c1 = (0.01*dynamic_range)**2
    c2 = (0.03*dynamic_range)**2
    filter_args = {'ddepth': -1,
                    'ksize': (win_size,win_size), 
                    'normalize': True,
                    'borderType': cv2.BORDER_REFLECT}

    ux = cv2.boxFilter(im1, **filter_args)
    uy = cv2.boxFilter(im2, **filter_args)

    uxx = cv2.boxFilter(im1*im1, **filter_args)
    uyy = cv2.boxFilter(im2*im2, **filter_args)
    uxy = cv2.boxFilter(im1*im2, **filter_args)
    
    vx = uxx - ux * ux
    vy = uyy - uy * uy
    vxy = uxy - ux * uy

    A1, A2, B1, B2 = ((2 * ux * uy + c1,
                    2 * vxy + c2,
                    ux ** 2 + uy ** 2 + c1,
                    vx + vy + c2))
        

    return (A1 * A2) / (B1 * B2)

def scalar_ssim(im1,im2, mask=None, win_size=7, dynamic_range = 1.0):
    if mask is None:
        mask = np.ones_like(im1)
    ssim_map = ssim(im1,im2, win_size, dynamic_range)
    masked_ssim_map = ssim_map*mask
    return np.sum(masked_ssim_map)/np.sum(mask)