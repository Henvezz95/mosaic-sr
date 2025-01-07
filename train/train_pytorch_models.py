import tensorflow as tf
from tqdm import tqdm

import numpy as np

import sys
sys.path.insert(0,'./')
sys.path.insert(0,'./3rd-party/')
from RT4KSR.arch import RT4KSR_Rep
from ECBSR.plainsr import PlainSR
from FMEN.test_fmen import TEST_FMEN
from reparameterize import reparameterize, reparameterize_ECBSR
from training_config_function import get_training_config
from utils.data_samplers import processed_batch_generator_real, generic_sampler

import torch
import os
import subprocess


def weighted_mse(input, target, weight):
    return torch.mean(weight * (input - target) ** 2)

def train_one_epoch(iterator_train, iterator_val, loss_function, steps_per_epoch, cuda=True, weighted_loss=False):
    train_loss = 0
    val_loss = 0
    val_steps = steps_per_epoch//5

    model.train(True)
    for _ in tqdm(range(steps_per_epoch)):
        x,y_true,weights,_ = next(iterator_train)
        optimizer.zero_grad()

        if cuda:
            outputs = model(torch.from_numpy(x).to(device))
            if weighted_loss:
                loss = weighted_mse(outputs, 
                                    torch.from_numpy(y_true).to(device), 
                                    torch.from_numpy(weights).to(device))
            else:
                loss = loss_function(outputs, 
                                    torch.from_numpy(y_true).to(device))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss = train_loss / steps_per_epoch
    
    model.eval()
    for i in range(val_steps):
        # Disable gradient computation during validation
        with torch.no_grad():
            x,y_true,weights,_ = next(iterator_val)
            outputs = model(torch.from_numpy(x).to(device))
            if weighted_loss:
                v_loss = weighted_mse(outputs, 
                                    torch.from_numpy(y_true).to(device), 
                                    torch.from_numpy(weights).to(device))
            else:
                v_loss = loss_function(outputs, torch.from_numpy(y_true).to(device))
            val_loss += v_loss.item()
    val_loss = val_loss / val_steps
    return train_loss, val_loss

def check_before_save(path, counter=0):
    extension = '.'+path.split('.')[-1]
    indexed_path = path[:-len(extension)]+'_'+str(counter)+extension
    while os.path.exists(indexed_path):
        counter+=1
        extension = '.'+path.split('.')[-1]
        indexed_path = path[:-len(extension)]+'_'+str(counter)+extension
    return indexed_path


# Set the path to the dataset here
tf_records_path = './TFDataset/'
shuffle_buffer_size = 100

train_path = tf_records_path+'Train/'
val_path = tf_records_path+'Validation/'

# Size of real crops. Synthetic Crops are always 96x96
H = 96
W = 96

network_types = ['eSR-TR_s2_K7_C16']
n_parallel_reads = 4
use_mask = False
device = torch.device("cuda:0")

seed=16

for gn in range(4):
    subprocess.run(["python3", 
                    "./buildTFDataset/create_dataset.py", 
                    "-c", "./configs/dataset_config.yaml", 
                    "-k", str(gn)])
    
for network_type in network_types:
    model, config, loss_function, optimizer, num_epochs, batch_size, scheduler = get_training_config(network_type, device)

    dataset_train = generic_sampler(train_path, n_parallel_reads=n_parallel_reads)
    dataset_t = processed_batch_generator_real(dataset_train, 
                                                H, W, 
                                                batch_size, 
                                                apply_contrast_stretch=True, 
                                                contrast_change=False,
                                                rotate=True, 
                                                apply_noise=False,
                                                ppe_min=1.0, ppe_max=1.6)
    dataset_val = generic_sampler(val_path, n_parallel_reads=n_parallel_reads)
    dataset_v = processed_batch_generator_real(dataset_val, 
                                                H, W, 
                                                batch_size, 
                                                apply_contrast_stretch=True, 
                                                contrast_change=False,
                                                rotate=True, 
                                                apply_noise=False,
                                                ppe_min=1.0, ppe_max=1.6)

    dataset_t = dataset_t.map(lambda x_LD, x_HD, x_mask, ppe: [ 
                            tf.transpose(x_LD, perm=[0,3,1,2]),
                            tf.transpose(x_HD, perm=[0,3,1,2]),
                            tf.transpose(x_mask, perm=[0,3,1,2]),
                            ppe])
    dataset_v = dataset_v.map(lambda x_LD, x_HD, x_mask, ppe: [
                            tf.transpose(x_LD, perm=[0,3,1,2]),
                            tf.transpose(x_HD, perm=[0,3,1,2]),
                            tf.transpose(x_mask, perm=[0,3,1,2]),
                            ppe])

    iterator_train = dataset_t.as_numpy_iterator()
    iterator_val = dataset_v.as_numpy_iterator()
    num_weights = sum(p.numel() for p in model.parameters())
    print('Model type:', network_type)
    print('Number of weights:', num_weights)

    for epoch in range(num_epochs):
        print('EPOCH {}:'.format(epoch + 1))
        print('Learning Rate=', optimizer.param_groups[0]['lr'])

        train_loss, val_loss = train_one_epoch(iterator_train, 
                                            iterator_val, 
                                            steps_per_epoch=int(np.ceil(3200/batch_size)), 
                                            loss_function=loss_function, 
                                            weighted_loss=use_mask)
        scheduler.step()
        print('Train Loss =', train_loss, 'Val Loss =', val_loss)

    if use_mask:
        footer = f'_mask_{str(gn)}'
    else:
        footer = f'_{str(gn)}'
    
    if network_type.split('_')[0] == 'rt4ksr':
        rep_config = {k:v for k,v in config.items()}
        rep_config['is_train'] = False
        rep_model = RT4KSR_Rep(**rep_config).to(device)
        rep_model = reparameterize(model, rep_model, device)
        torch.save(rep_model, f'./PT_models/{network_type}_{footer}.pt')
    elif network_type =='fmen':
        rep_model = TEST_FMEN(config).to(device)
        rep_model = reparameterize(model, rep_model, device)
        torch.save(rep_model, f'./PT_models/{network_type}_{footer}.pt')
    elif network_type.split('_')[0] == 'ecbsr':
        rep_config = {k:v for k,v in config.items() if k != 'with_idt'}
        rep_model = PlainSR(**rep_config) 
        rep_model = reparameterize_ECBSR(model, rep_model)
        torch.save(rep_model, f'./PT_models/{network_type}_{footer}.pt')
    else:
        torch.save(model, f'./PT_models/{network_type}_{footer}.pt')