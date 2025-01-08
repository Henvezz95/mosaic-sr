import tensorflow as tf
import numpy as np
import yaml
import torch

import sys
sys.path.insert(0,'./')
sys.path.insert(0,'./3rd-party/')
from utils.utils import iterativeUpscale
from utils.converters import iterative2tflite
from utils.data_samplers import generic_sampler, processed_batch_generator_real
from time import perf_counter_ns
from tqdm import tqdm

def list2str(mylist):
    return'['+",".join([str(x) for x in mylist])+']'

tf_records_path = '../TFDataset_real/'
test_path = tf_records_path+'Validation/'
batch_size=1

# Size of real crops. Synthetic Crops are always 96x96
H_real = 128
W_real = 128
NUM_REPEATS = 5
selected_ppe = 1.3

model_path = '../super-resolution-gan/TF_Models/aggregate_model_s0.0-r1.0_2_3_3_ck_cl_0.h5'
th1 = [0.0, 0.05, 0.07, 0.09, 0.1, 0.11, 0.13, 0.15, 0.18, 1.0]
th2 = [0.0, 0.05, 0.07, 0.09, 0.1, 0.11, 0.13, 0.15, 0.18, 1.0]
thresholds_grid = np.meshgrid(th1, th2)
thresholds_zipped = np.array(list(zip(thresholds_grid[0].ravel(), thresholds_grid[1].ravel())))
thresholds = list(filter(lambda x: x[0]<=x[1],thresholds_zipped))
thresholds = [x.tolist() for x in thresholds]

g_model = tf.keras.models.load_model(model_path)
interpreters = iterative2tflite(g_model, input_shape=None, num_threads=1)
output_dictionary = {'nn_times':{}, 'total_times':{}}
    
for th_it in thresholds:
    th = np.power(th_it,2)
    dataset_test = generic_sampler(test_path, n_parallel_reads=4, repeat=False)
    dataset_test = processed_batch_generator_real(dataset_test, 
                                                H_real, W_real, 
                                                batch_size, 
                                                apply_contrast_stretch=True, 
                                                contrast_change=False,
                                                rotate=False, 
                                                apply_noise=False,
                                                ppe_min=selected_ppe, ppe_max=selected_ppe)
    
    iterator = dataset_test.as_numpy_iterator()
    np.random.seed(0)
    all_times = []
    total_nn_times = []
    for it in tqdm(iterator):
        img_LD, img_HD, img_mask, ppe = it
        img_LD = np.float32(img_LD)
        current_times = []
        nn_times = []
        for i in range(NUM_REPEATS):
            with tf.device('/cpu:0'):
                output_data, total_nn_time, total_data_time = iterativeUpscale(img_LD[0,:,:,0], interpreters, th)
                nn_times.append(total_nn_time)
                current_times.append(total_nn_time + total_data_time)
        all_times.append(min(current_times))
        total_nn_times.append(min(nn_times))
    output_dictionary['nn_times'][list2str(th_it)] = np.mean(total_nn_times).tolist()
    output_dictionary['total_times'][list2str(th_it)] = np.mean(all_times).tolist()


    with open('./results/times.yaml', 'w') as outfile:
        yaml.dump(output_dictionary, outfile, default_flow_style=False, sort_keys=False)