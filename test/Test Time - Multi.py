import tensorflow as tf
import numpy as np
import yaml
import torch

import sys
sys.path.insert(0,'./')
sys.path.insert(0,'./3rd-party/')
from AsConvSR.AsConvSR import AsConvSR
from utils.converters import pytorch2tflite, keras2tflite
from utils.data_samplers import generic_sampler, processed_batch_generator_real
from time import perf_counter_ns
from tqdm import tqdm

tf_records_path = '../TFDataset_real/'
test_path = tf_records_path+'Validation/'
batch_size=1

# Size of real crops. Synthetic Crops are always 96x96
H_real = 128
W_real = 128
NUM_REPEATS = 5
selected_ppe = 1.3

folder = '../super-resolution-gan/PT_Models/'
models = ['fmen',
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

footer = '_s0.0-r1.0'
model_type = 'torch'

output_dictionary = {}
    
for model in models:
    print('Model:', model)
    model_path = folder+model+footer+'_0'
    if model_type == 'tf':
        g_model = tf.keras.models.load_model(model_path+'.h5')
        interpreter = keras2tflite(g_model, input_shape=(batch_size,H_real,W_real,1), num_threads=1)
    elif model_type == 'torch':
        if model == 'asconvsr':
            g_model = AsConvSR(in_ch=1, out_ch=1, scale_factor=2, device=torch.device('cpu'))
            g_model_load = torch.load(model_path+'.pt', map_location=torch.device('cpu'))
            loaded_state_dict = g_model_load.state_dict()
            for k,v in loaded_state_dict.items():
                loaded_state_dict[k] = v.to(torch.device('cpu'))
            g_model.load_state_dict(loaded_state_dict)
        else:
            g_model = torch.load(model_path+'.pt', map_location=torch.device('cpu'))
        g_model.eval()
        interpreter = pytorch2tflite(g_model, tmp_folder='./tmp/', input_shape=(batch_size,1,H_real,W_real), num_threads=1)
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
        for it in tqdm(iterator):
            img_LD, img_HD, img_mask, ppe = it
            img_LD = np.float32(img_LD)
            if model_type == 'torch':
                img_LD = img_LD[:,np.newaxis,:,:,0]
            current_times = []
            for i in range(NUM_REPEATS):
                with tf.device('/cpu:0'):
                    # Allocate tensors
                    interpreter.allocate_tensors()
                    # Get input and output tensors
                    input_details = interpreter.get_input_details()
                    output_details = interpreter.get_output_details()
                    start = perf_counter_ns()
                    # Test the model on random input data
                    input_shape = input_details[0]['shape']
                    interpreter.set_tensor(input_details[0]['index'], img_LD)
                    interpreter.invoke()
                    output_data = interpreter.get_tensor(output_details[0]['index'])
                    current_times.append((perf_counter_ns()-start)/1e6)
            all_times.append(min(current_times))
        output_dictionary[model] = np.mean(all_times).tolist()


    with open('./results/times.yaml', 'w') as outfile:
        yaml.dump(output_dictionary, outfile, default_flow_style=False, sort_keys=False)