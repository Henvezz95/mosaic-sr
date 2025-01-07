import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau
from wrapping_models import simple_model, iterative_model
from model_building_functions import add_network, starting_model

import sys
sys.path.insert(0,'./')
from utils.utils import pick_gpu_lowest_memory
from utils.data_samplers import processed_batch_generator_real, generic_sampler
from utils.reparam_blocks import Conv2D_CK7x7, Conv2D_CL, convertWeightsCK, convertWeightsCL


# Set Up the GPU that is available
os.environ["CUDA_VISIBLE_DEVICES"] = str(pick_gpu_lowest_memory())
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)

config = tf.compat.v1.ConfigProto()
session =tf.compat.v1.InteractiveSession(config=config)

# Set the path to the dataset here
tf_records_path = './TFDataset/'
shuffle_buffer_size = 100
batch_size = 16
seed = 16
n_parallel_reads = 4
use_mask = False

train_path = tf_records_path+'Train/'
val_path = tf_records_path+'Validation/'

# Size of real crops
H = 96
W = 96

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
    

rlrop = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=40, min_lr=0.00005)
ker_size = 7
alpha=0.1
to_rep_ck = False
to_rep_cl = False
batch_norm = False
epochs = 200
epochs_tuning = 30
ch = [[16,16], 
      [32,32,16], 
      [96,64,32]]
ch_1, ch_2, ch_3 = ch

model_torep, x1_torep = starting_model(ch = ch_1, 
                             alpha=alpha, 
                             ker_size=ker_size, 
                             batch_norm = batch_norm, 
                             to_rep_ck=to_rep_ck, 
                             to_rep_cl=to_rep_cl)

model, x1 = starting_model(ch = ch_1, 
                       alpha=alpha, 
                       ker_size=ker_size, 
                       batch_norm = batch_norm, 
                       to_rep_ck=False, 
                       to_rep_cl=False)

out1_torep = model_torep.output
out1 = model.output
train_model = simple_model(model_torep, use_mask=use_mask)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
train_model.compile(optimizer=optimizer, run_eagerly=True)
train_model.fit(dataset_t, epochs=epochs, steps_per_epoch=200, callbacks=[rlrop])

for layer in model.layers:
    layer.trainable = False
train_model.trainable = False
train_model.compile(optimizer=optimizer, run_eagerly=True)
out2_torep, x2_torep = add_network(model_torep.input, 
                                   x1_torep, 
                                   ker_size, 
                                   ch_2, 
                                   iter_index=2, 
                                   alpha=alpha, 
                                   batch_norm=batch_norm,
                                   to_rep_ck=to_rep_ck, 
                                   to_rep_cl=to_rep_cl)
model_torep = Model(model_torep.input, out2_torep)
out2, x2 = add_network(model.input, 
                       x1, 
                       ker_size, 
                       ch_2, 
                       iter_index=2, 
                       alpha=alpha, 
                       batch_norm=batch_norm,
                       to_rep_ck=False, 
                       to_rep_cl=False)
model = Model(model.input, out2)

train_model = simple_model(model_torep, use_mask=use_mask)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
train_model.compile(optimizer=optimizer, run_eagerly=True)
train_model.fit(dataset_t, epochs=epochs, steps_per_epoch=200, callbacks=[rlrop])

for layer in model.layers:
    layer.trainable = False
train_model.trainable = False
train_model.compile(optimizer=optimizer, run_eagerly=True)
out3_torep, x3_torep = add_network(model_torep.input, 
                                   x2_torep, 
                                   ker_size, 
                                   ch_3, 
                                   iter_index=3, 
                                   alpha=alpha, 
                                   batch_norm=batch_norm,
                                   to_rep_ck=to_rep_ck, 
                                   to_rep_cl=to_rep_cl)
model_torep = Model(model_torep.input, out3_torep)
out3, x3 = add_network(model.input, 
                       x2, 
                       ker_size, 
                       ch_3, 
                       iter_index=3, 
                       alpha=alpha, 
                       batch_norm=batch_norm,
                       to_rep_ck=False, 
                       to_rep_cl=False)
model = Model(model.input, out3)
train_model = simple_model(model_torep, use_mask=use_mask, var_pred=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
train_model.compile(optimizer=optimizer, run_eagerly=True)
train_model.fit(dataset_t, epochs=epochs, steps_per_epoch=200, callbacks=[rlrop])

for layer in model.layers:
    layer.trainable = True
train_model.trainable = True
train_model.compile(optimizer=optimizer, run_eagerly=True)
aggregate_model_torep = Model(model_torep.input, [out1_torep,out2_torep,out3_torep])
aggregate_model = Model(model.input, [out1,out2,out3])

rlrop_tuning = ReduceLROnPlateau(monitor='total_loss', factor=0.5, patience=20, min_lr=1e-6)
aggregate_model_train = iterative_model(aggregate_model_torep, use_mask=use_mask)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
aggregate_model_train.compile(optimizer=optimizer, run_eagerly=True)
aggregate_model_train.fit(dataset_t, epochs=epochs, steps_per_epoch=200, callbacks=[rlrop_tuning])

########################################################################################
##############################  Model Reparameterization  ##############################
########################################################################################

for idx, layer in enumerate(aggregate_model_torep.layers):
    weights_torep = layer.get_weights()
    if isinstance(layer, Conv2D_CK7x7):
        weights_torep = [w.astype('float64') for w in weights_torep]
        new_weights = convertWeightsCK(weights_torep)
        new_weights = [n.astype('float32') for n in new_weights]
        aggregate_model.layers[idx].set_weights(new_weights)
    elif isinstance(layer, Conv2D_CL):
        weights_torep = [w.astype('float64') for w in weights_torep]
        new_weights = convertWeightsCL(weights_torep)
        new_weights = [n.astype('float32') for n in new_weights]
        aggregate_model.layers[idx].set_weights(new_weights)
    else:
        aggregate_model.layers[idx].set_weights(weights_torep)

#Save Aggregate Model
name = f'../TF_Models/aggregate_model_{len(ch[0])}_{len(ch[1])}_{len(ch[2])}'
if to_rep_ck:
    name = name+'_ck'
if to_rep_cl:
    name = name+'_cl'
aggregate_model.save(name+'_0.h5')