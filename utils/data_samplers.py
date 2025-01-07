import tensorflow as tf
from tensorflow.data import TFRecordDataset 
import cv2
import numpy as np
from glob import glob
from utils.tf_augmentations import random_rotate_uniform
from utils.tf_augmentations import change_contrast, blur, resize, add_noise

def parse_record_real(record):
    name_to_features = {
        'img': tf.io.FixedLenFeature([], tf.string),
        'shape': tf.io.FixedLenFeature((2), tf.int64),
        'polygon': tf.io.FixedLenFeature((8), tf.int64),
        'file_name': tf.io.FixedLenFeature([], tf.string),
        'ppe':   tf.io.FixedLenFeature((1), tf.float32),
    }
    return tf.io.parse_single_example(record, name_to_features)

def decode_record_real(record):
    shape = record['shape'][:2]
    img = tf.cast(
       tf.io.decode_png(record['img'], channels=0, dtype=tf.dtypes.uint8, name=None),
        tf.uint8)
    img = tf.reshape(img, shape)[:,:,tf.newaxis]
    polygon = tf.reshape(record['polygon'], (4,2))
    ppe = record['ppe']
    return img[:,:,0], polygon, ppe

def generic_sampler(dataset_path, n_parallel_reads, repeat = True):
    records = glob(dataset_path+'*.tfrecord')
    if repeat:
        return TFRecordDataset(records, num_parallel_reads=n_parallel_reads).repeat()
    else:
        return TFRecordDataset(records, num_parallel_reads=n_parallel_reads)
    

def processed_batch_generator_real(dataset_train, 
                                   H_window=96, 
                                   W_window=96, 
                                   batch_size=32, 
                                   apply_contrast_stretch=True, 
                                   rotate=True,
                                   contrast_change=True,
                                   apply_noise=True,
                                   ppe_min=1.0, ppe_max=1.6):
    fixed_resize = lambda img_LD, img_HD, polygon, ppe: resize(img_LD, 
                                                                        img_HD, 
                                                                        polygon, 
                                                                        ppe, 
                                                                        H_window=H_window, 
                                                                        W_window=W_window,
                                                                        ppe_min=ppe_min, 
                                                                        ppe_max=ppe_max)
    noise_function = lambda img_LD, img_HD, polygon, ppe: add_noise(img_LD, 
                                                                             img_HD, 
                                                                             polygon, 
                                                                             ppe, 
                                                                             apply_contrast_stretch=apply_contrast_stretch,
                                                                             bypass_noise = not(apply_noise))

    dataset_train = dataset_train.map(lambda x: parse_record_real(x))
    dataset_train = dataset_train.map(lambda x: decode_record_real(x))
    if rotate:
        dataset_train = dataset_train.map(
            lambda img, polygon, ppe: tf.py_function(func=random_rotate_uniform, 
                                                    inp=[img, polygon, ppe], 
                                                    Tout=[tf.uint8, tf.float32, tf.float32]), 
                                                    num_parallel_calls=4)
    if contrast_change:
        dataset_train = dataset_train.map(
            lambda img, polygon, ppe: tf.py_function(func=change_contrast, 
                                                    inp=[img, polygon, ppe], 
                                                    Tout=[tf.float32]*3), 
                                                    num_parallel_calls=4)
    else:
        dataset_train = dataset_train.map(lambda img, polygon, ppe: (tf.cast(img, tf.float32), tf.cast(polygon, tf.float32), tf.cast(ppe, tf.float32)))

    dataset_train = dataset_train.map(
        lambda img, polygon, ppe: tf.py_function(func=blur, 
                                                          inp=[img, polygon, ppe], 
                                                          Tout=[tf.float32]*4), 
                                                          num_parallel_calls=4)
    dataset_train = dataset_train.map(
        lambda img_LD, img_HD, polygon, ppe: tf.py_function(func=fixed_resize, 
                                                                     inp=[img_LD, img_HD, polygon, ppe], 
                                                                     Tout=[tf.float32]*4), 
                                                                     num_parallel_calls=4)
    dataset_train = dataset_train.map(
        lambda img_LD, img_HD, polygon, ppe: tf.py_function(func=noise_function, 
                                                                     inp=[img_LD, img_HD, polygon, ppe], 
                                                                     Tout=[tf.float32]*4), 
                                                                     num_parallel_calls=4)


    dataset_t = dataset_train.batch(batch_size)
    dataset_t = dataset_t.prefetch(10)
    return dataset_t