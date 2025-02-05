import tensorflow as tf
import numpy as np
from tqdm import tqdm
from pyzbar.pyzbar import decode
import yaml
from copy import deepcopy

import sys
import subprocess
sys.path.insert(0,'./')
sys.path.insert(0,'./3rd-party/')
from utils.metrics import  masked_psnr, scalar_ssim
from utils.utils import iterativeUpscale, fast_clip
from utils.converters import iterative2tflite
from utils.data_samplers import generic_sampler, processed_batch_generator_real
from utils.readers import zbar_reader

def list2str(mylist):
    return'['+",".join([str(x) for x in mylist])+']'

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session =tf.compat.v1.InteractiveSession(config=config)

tf_records_path_synth = '../TFDataset_real/'
test_path = tf_records_path_synth+'Validation/'
batch_size=128

# Size of image crops
H_real = 128
W_real = 128
border = 3
input_size = [7,7]
dataset_config_file = "./configs/dataset_config.yaml"


folder = '../super-resolution-gan/TF_Models/'
model_name  = 'aggregate_model_7x7_s0.0-r1.0_2_3_3_ck_cl'
th1 = [0.0, 0.05, 0.07, 0.09, 0.1, 0.11, 0.13, 0.15, 0.18, 1.0]
th2 = [0.0, 0.05, 0.07, 0.09, 0.1, 0.11, 0.13, 0.15, 0.18, 1.0]
thresholds_grid = np.meshgrid(th1, th2)
thresholds_zipped = np.array(list(zip(thresholds_grid[0].ravel(), thresholds_grid[1].ravel())))
thresholds = list(filter(lambda x: x[0]<=x[1],thresholds_zipped))
thresholds = [x.tolist() for x in thresholds]

results = {'psnr_all':{}, 'psnr_mask':{},'ssim_all':{}, 'ssim_mask':{}, 'zbar':{}}

zbar_decoder = zbar_reader()
decode_image = True

with open(dataset_config_file) as yaml_file:
    dataset_config = yaml.safe_load(yaml_file)

k_folds = dataset_config['k_folds']

for i in range(k_folds):
    subprocess.run(["python3", 
                    "./buildTFDataset/create_dataset.py", 
                    "-c", dataset_config_file, 
                    "-k", str(i)])
    
    model_path = folder+model_name+'_'+str(i)
    g_model = tf.keras.models.load_model(model_path+'.h5')
    interpreters = iterative2tflite(g_model, input_shape=None, num_threads=1)
    
    for th in thresholds:
        th_str = list2str(th)
        if th_str not in results['psnr_mask']:
            results['psnr_mask'][th_str] = np.zeros((7,k_folds)).tolist()
            results['psnr_all'][th_str] = np.zeros((7,k_folds)).tolist()
            results['ssim_mask'][th_str] = np.zeros((7,k_folds)).tolist()
            results['ssim_all'][th_str] = np.zeros((7,k_folds)).tolist()
            results['zbar'][th_str] = np.zeros((7,k_folds), dtype=np.int32).tolist()
        for j in tqdm(range(7)):
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
            np.random.seed(0)

            for it in tqdm(iterator):
                img_LD, img_HD, img_mask, ppe = it
                img_LD = np.float32(img_LD)
                img_HD = np.float32(img_HD)
                img_mask = np.float32(img_mask)

                imgs_up = np.zeros_like(img_HD)
                for k in range(len(img_HD)):
                    img_up, _, _ = iterativeUpscale(img_LD[k,:,:,0], interpreters, np.power(th,2), ker=input_size)
                    imgs_up[k] = img_up[:,:,np.newaxis]
                imgs_up = imgs_up[:,:,:,0]

                if border > 0:
                    imgs_up = imgs_up[:,border*2:-border*2,border*2:-border*2]
                    img_HD = img_HD[:,border*2:-border*2,border*2:-border*2]
                    img_mask = img_mask[:,border*2:-border*2,border*2:-border*2]
                for k in range(len(img_LD)):
                    psnr_mask_results_sr.append(masked_psnr(imgs_up[k], img_HD[k,:,:,0], 
                                                            np.round(img_mask[k,:,:,0]), R=1))
                    psnr_results_sr.append(masked_psnr(imgs_up[k], img_HD[k,:,:,0], R=1))
                    ssim_mask_results_sr.append(scalar_ssim(imgs_up[k], img_HD[k,:,:,0], 
                                                            np.round(img_mask[k,:,:,0])))
                    ssim_results_sr.append(scalar_ssim(imgs_up[k], img_HD[k,:,:,0]))
                    img2decode = deepcopy(imgs_up[k])
                    img2decode = np.round(fast_clip(img2decode*255,0,255)).astype('uint8')
                    if decode_image:
                        zbar_decoder.work(img2decode)
                        if(len(zbar_decoder.get_decoded_results())==1):
                            results['zbar'][th_str][j][i]+=1

            results['psnr_mask'][th_str][j][i] = np.mean(psnr_mask_results_sr).tolist()
            results['psnr_all'][th_str][j][i] = np.mean(psnr_results_sr).tolist()
            results['ssim_mask'][th_str][j][i] = np.mean(ssim_mask_results_sr).tolist()
            results['ssim_all'][th_str][j][i] = np.mean(ssim_results_sr).tolist()

    with open('./results/results_iterative_7x7.yaml', 'w') as outfile:
        yaml.dump(results, outfile, default_flow_style=False, sort_keys=False)