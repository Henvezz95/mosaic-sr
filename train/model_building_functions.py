import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, LeakyReLU, Conv2DTranspose, Input, Concatenate, BatchNormalization

import sys
sys.path.insert(0,'./')
from utils.reparam_blocks import Conv2D_CK7x7, Conv2D_CL

def add_network(img_input, 
                embedding_input, 
                ker_size, 
                channels_per_layer, 
                batch_norm=True, 
                to_rep_ck=True, 
                to_rep_cl=True, 
                variance_prediction=True, 
                iter_index = 0, 
                alpha=0.1):
    current_index = 1
    if to_rep_ck:
        x = Conv2D_CK7x7(channels_per_layer[0], (ker_size, ker_size), 
                      name=f'Conv_{str(iter_index)}_{current_index}')(img_input)
    else:
        x = Conv2D(channels_per_layer[0], (ker_size, ker_size), 
                   name=f'Conv_{str(iter_index)}_{current_index}', padding='same')(img_input)
    x = LeakyReLU(alpha=alpha, name=f'Act_{str(iter_index)}_{current_index}')(x)
    x = Concatenate(name=f'Concat_{str(iter_index)}_1')([embedding_input, x])
    current_index += 1
    for i in range(1, len(channels_per_layer)):
        if to_rep_cl:
            x = Conv2D_CL(channels_per_layer[i], ratio=2, name=f'Conv_{str(iter_index)}_{current_index}')(x)
        else:
            x = Conv2D(channels_per_layer[i], (1,1), padding='same', name=f'Conv_{str(iter_index)}_{current_index}')(x)
        if batch_norm:
            x = BatchNormalization(momentum=0.9, name=f'BN_{str(iter_index)}_{current_index}')(x)
        x = LeakyReLU(alpha=alpha, name=f'Act_{str(iter_index)}_{current_index}')(x)
        current_index += 1
    embedding_output = x
    mu_out = Conv2DTranspose(1, (2,2), strides=(2,2), padding='same', activation = 'sigmoid', name=f'ConvT_{str(iter_index)}')(x)
    
    if variance_prediction:
        sigma_out = Conv2D(1, (1,1), padding='same', activation='sigmoid', name= f'ConvSig_{str(iter_index)}')(x)
        sigma_out = tf.keras.layers.UpSampling2D(size=(2, 2), name= f'Up_{str(iter_index)}')(sigma_out)
        img_output = Concatenate(name=f'Concat_{str(iter_index)}_2')([mu_out, sigma_out])
    else:
        img_output = mu_out
    
    return img_output, embedding_output

def starting_model(input_shape =(None,None,1), 
                   ch = [16,16], 
                   alpha=0.1, 
                   ker_size=7, 
                   batch_norm = True, 
                   to_rep_ck=True, 
                   to_rep_cl=True):
    img_input = Input(input_shape, name='Model_input')
    if to_rep_ck:
        x = Conv2D_CK7x7(ch[0], (ker_size, ker_size), name=f'Conv_1_1')(img_input)
    else:
        x = Conv2D(ch[0], (ker_size, ker_size), name=f'Conv_1_1', padding='same')(img_input)
    if batch_norm:
        x = BatchNormalization(momentum=0.9, name=f'BN_1_1')(x)
    x = LeakyReLU(alpha=alpha, name=f'Act_1_1')(x)

    for i in range(1, len(ch)):
        if to_rep_cl:
            x = Conv2D_CL(ch[i], ratio=4, name=f'Conv_1_{i+1}')(x)
        else:
            x = Conv2D(ch[i], (1,1), padding='same', name=f'Conv_1_{i+1}')(x)
        if batch_norm:
            x = BatchNormalization(momentum=0.9, name=f'BN_1_{i+1}')(x)
        x = LeakyReLU(alpha=alpha, name=f'Act_1_{i+1}')(x)

    mu_out = Conv2DTranspose(1, (2,2), strides=(2,2), padding='same',  activation = 'sigmoid', name=f'ConvT_1')(x)
    sigma_out = Conv2D(1, (1,1), padding='same', activation='sigmoid', name=f'ConvSig_1')(x)
    sigma_out = tf.keras.layers.UpSampling2D(size=(2, 2), name=f'Up_1')(sigma_out)
    out1 = Concatenate(name='Concat_1_2')([mu_out, sigma_out])
    model = Model(img_input, out1)
    return model, x