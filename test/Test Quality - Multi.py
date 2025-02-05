import tensorflow as tf
import numpy as np
from tqdm import tqdm
import yaml
import torch

import sys
import cv2
import subprocess
from copy import deepcopy
sys.path.insert(0,'./')
sys.path.insert(0,'./3rd-party/')
from AsConvSR.AsConvSR import AsConvSR
from utils.utils import fast_clip, interpolations
from utils.metrics import masked_psnr, scalar_ssim
from utils.readers import zbar_reader
from utils.data_samplers import generic_sampler, processed_batch_generator_real

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session =tf.compat.v1.InteractiveSession(config=config)

tf_records_path = '../TFDataset_real/'
test_path = tf_records_path+'Validation/'
batch_size = 32

# Size of real crops. Synthetic Crops are always 96x96
H_real = 128
W_real = 128
border = 3
decode_image = True

folder = '../super-resolution-gan/PT_Models/'
models = ['asconvsr', 
          'fmen',
          'rfdn', 
          'imdn', 
          'rt4ksr_xxs',
          'rt4ksr_s', 
          'rt4ksr_xl', 
          'fsrcnn', 
          'espcn', 
          'ecbsr_M4C8', 
          'imdn_rtc', 
          'eSR-MAX_s2_K5_C8', 
          'eSR-TM_s2_K7_C16', 
          'eSR-TR_s2_K7_C16']

#models = ['cubic', 'lanczos', 'linear']
footer = '_s0.0-r1.0'
model_type = 'torch'
device = torch.device("cuda:0")

output_dictionary = {'psnr_all':{}, 'psnr_mask':{},'ssim_all':{}, 'ssim_mask':{}, 'zbar':{}}
zbar_decodings_ld = [[0]*4,[0]*4,[0]*4,[0]*4,[0]*4,[0]*4,[0]*4]
ratio = []

zbar_decoder = zbar_reader()

for i in range(4):
    subprocess.run(["python3", 
                    "./buildTFDataset/create_dataset.py", 
                    "-c", "./configs/dataset_config.yaml", 
                    "-k", str(i)])
    
    for idx, model in enumerate(models):
        if model not in output_dictionary['psnr_all']:
            output_dictionary['psnr_all'][model] = [[0]*4,[0]*4,[0]*4,[0]*4,[0]*4,[0]*4,[0]*4]
            output_dictionary['psnr_mask'][model] = [[0]*4,[0]*4,[0]*4,[0]*4,[0]*4,[0]*4,[0]*4]
            output_dictionary['ssim_all'][model] = [[0]*4,[0]*4,[0]*4,[0]*4,[0]*4,[0]*4,[0]*4]
            output_dictionary['ssim_mask'][model] = [[0]*4,[0]*4,[0]*4,[0]*4,[0]*4,[0]*4,[0]*4]
            output_dictionary['zbar'][model] = [[0]*4,[0]*4,[0]*4,[0]*4,[0]*4,[0]*4,[0]*4]
        
        model_path = folder+model+footer+'_'+str(i)
        if model_type == 'tf' or model_type == 'iterative':
            g_model = tf.keras.models.load_model(model_path+'.h5')
        elif model_type == 'torch':
            if model == 'asconvsr':
                g_model = AsConvSR(in_ch=1, out_ch=1, scale_factor=2, device=torch.device('cpu'))
                g_model_load = torch.load(model_path+'.pt', map_location=torch.device('cpu'))
                loaded_state_dict = g_model_load.state_dict()
                for k,v in loaded_state_dict.items():
                    loaded_state_dict[k] = v.to(torch.device('cpu'))
                g_model.load_state_dict(loaded_state_dict)
            else:
                g_model = torch.load(model_path+'.pt').to(device)
            g_model.eval()

        for j in tqdm(range(7)):
            np.random.seed(0)
            dataset_test = generic_sampler(test_path, n_parallel_reads=4, repeat=False)
            dataset_test = processed_batch_generator_real(dataset_test, 
                                                        H_real+border*2, W_real+border*2, 
                                                        batch_size, 
                                                        apply_contrast_stretch=True, 
                                                        contrast_change=False,
                                                        rotate=False, 
                                                        apply_noise=False,
                                                        ppe_min=1.0+j/10, ppe_max=1.0+j/10)
            iterator = dataset_test.as_numpy_iterator()

            psnr_mask_results_sr = []
            psnr_results_sr = []
            ssim_mask_results_sr = []
            ssim_results_sr = []
            zbar_tot = []

            for it in iterator:
                img_LD, img_HD, img_mask, ppe = it
                img_LD = np.float32(img_LD)
                img_HD = np.float32(img_HD)
                img_mask = np.float32(img_mask)

                if model_type == 'tf':
                    imgs_up = g_model.predict(img_LD, verbose=0)[:,:,:,0]
                elif model_type == 'torch':
                    if model == 'asconvsr':
                        torch_input = torch.from_numpy(img_LD[:,np.newaxis,:,:,0]).to(torch.device('cpu'))
                        torch_output = g_model(torch_input).detach().cpu().numpy()
                    else:
                        g_model = g_model.to(device)
                        torch_input = torch.from_numpy(img_LD[:,np.newaxis,:,:,0]).to(device)
                        torch_output = g_model(torch_input).detach().cpu().numpy()
                    imgs_up = torch_output[:,0]
                elif model in interpolations:
                    imgs_up = np.empty((len(img_HD), (W_real+border*2)*2, (H_real+border*2)*2), 
                                       dtype=np.float32)
                    
                    for idx in range(len(imgs_up)):
                        imgs_up[idx] = cv2.resize(img_LD[idx], 
                                                  ((W_real+border*2)*2, (H_real+border*2)*2),
                                                  interpolation=interpolations[model])

                if border > 0:
                    imgs_up = imgs_up[:,border*2:-border*2,border*2:-border*2]
                    img_HD = img_HD[:,border*2:-border*2,border*2:-border*2]
                    img_mask = img_mask[:,border*2:-border*2,border*2:-border*2]
                for k in range(len(img_LD)):
                    ratio.append(np.mean(img_mask[k,:,:,0]))
                    psnr_mask_results_sr.append(masked_psnr(imgs_up[k], img_HD[k,:,:,0], np.round(img_mask[k,:,:,0]), R=1))
                    psnr_results_sr.append(masked_psnr(imgs_up[k], img_HD[k,:,:,0], R=1))
                    ssim_mask_results_sr.append(scalar_ssim(imgs_up[k], img_HD[k,:,:,0], np.round(img_mask[k,:,:,0])))
                    ssim_results_sr.append(scalar_ssim(imgs_up[k], img_HD[k,:,:,0]))
                    if decode_image:
                        img2decode = deepcopy(imgs_up[k])
                        img2decode = np.round(fast_clip(img2decode*255,0,255)).astype('uint8')
                        zbar_decoder.work(img2decode)
                        if(len(zbar_decoder.get_decoded_results())==1):
                            output_dictionary['zbar'][model][j][i]+=1
                        if idx == 0:
                            img2decode = deepcopy(img_LD[k,:,:,0])
                            img2decode = np.round(fast_clip(img2decode*255,0,255)).astype('uint8')
                            zbar_decoder.work(img2decode)
                            if(len(zbar_decoder.get_decoded_results())==1):
                                zbar_decodings_ld[j][i]+=1

                    
            output_dictionary['psnr_mask'][model][j][i] = np.mean(psnr_mask_results_sr).tolist()
            output_dictionary['psnr_all'][model][j][i] = np.mean(psnr_results_sr).tolist()
            output_dictionary['ssim_mask'][model][j][i] = np.mean(ssim_mask_results_sr).tolist()
            output_dictionary['ssim_all'][model][j][i] = np.mean(ssim_results_sr).tolist()
            output_dictionary['zbar_ld'] = zbar_decodings_ld
            output_dictionary['mask_ratio'] = np.mean(ratio)


    with open('./results/results_interpolations.yaml', 'w') as outfile:
        yaml.dump(output_dictionary, outfile, default_flow_style=False, sort_keys=False)