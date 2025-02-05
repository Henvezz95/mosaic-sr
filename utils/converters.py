import torch
import tensorflow as tf
import tf2onnx
import onnx
from onnx_tf.backend import prepare

import tensorflow as tf 
from tensorflow.keras.models import  Model
from tensorflow.keras.layers import  Input, BatchNormalization, Dense, Concatenate
import numpy as np


def keras2tflite(model, input_shape=None, num_threads=1):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable LiteRT ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    tflite_model = converter.convert()
    interpreter = tf.lite.Interpreter(model_content=tflite_model, num_threads=num_threads)
    if input_shape is not None:
        interpreter.resize_tensor_input(0, input_shape, strict=True)
    return interpreter

def pytorch2tflite(model, tmp_folder='./tmp/', input_shape=(1,1,96,96), num_threads=1):
    sample_input = torch.rand(input_shape)
    torch.onnx.export(
        model,                  # PyTorch Model
        sample_input,                    # Input tensor
        './tmp/tmp.onnx',        # Output file (eg. 'output_model.onnx')
        opset_version=12,       # Operator support version
        input_names=['input'],   # Input tensor name (arbitary)
        output_names=['output'] # Output tensor name (arbitary)
    )
    onnx_model = onnx.load(f"{tmp_folder}tmp.onnx")
    onnx.checker.check_model(onnx_model)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(f'{tmp_folder}tmp.pb')
    converter = tf.lite.TFLiteConverter.from_saved_model(f'{tmp_folder}tmp.pb')
    converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # enable LiteRT ops.
    tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    tflite_model = converter.convert()

    # Save the model
    with open(f'{tmp_folder}tmp.tflite', 'wb') as f:
        f.write(tflite_model)

    interpreter = tf.lite.Interpreter(model_path=f"{tmp_folder}tmp.tflite", num_threads=num_threads)
    return interpreter

def pytorch2onnx(model, tmp_folder='./tmp/', input_shape=(1,1,96,96)):
    sample_input = torch.rand(input_shape)
    torch.onnx.export(
        model,                  # PyTorch Model
        sample_input,                    # Input tensor
        './tmp/tmp.onnx',        # Output file (eg. 'output_model.onnx')
        opset_version=13,       # Operator support version
        input_names=['input'],   # Input tensor name (arbitary)
        output_names=['output'], # Output tensor name (arbitary)
        dynamic_axes={
            'input': {
                0: 'batch_size',  # make batch size dynamic
                2: 'height',      # make height dynamic
                3: 'width'        # make width dynamic
            },
            'output': {
                0: 'batch_size',  # match batch dimension in output
                2: 'height',      # match height dimension in output
                3: 'width'        # match width dimension in output
            }
        }
    )
    onnx_model = onnx.load(f"{tmp_folder}tmp.onnx")
    onnx.checker.check_model(onnx_model)
    return onnx_model


def keras2onnx(model, 
               save_path='./tmp/tmp.onnx', 
               input_shape=(None, None, None, 1), 
               opset=13):

    input_signature = [tf.TensorSpec(shape=input_shape, dtype=tf.float32, name='input')]

    onnx_model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=opset,
        output_path=save_path
    )

    loaded_onnx_model = onnx.load(save_path)
    onnx.checker.check_model(loaded_onnx_model)
    return loaded_onnx_model


def get_aggregate_model_stats(model):
    num_iter = 0
    channel_counter = {}
    for layer in model.layers:
        name = layer.name
        splitted_name = name.split('_')
        if splitted_name[0] not in channel_counter:
            channel_counter[splitted_name[0]] = 0
        channel_counter[splitted_name[0]] += 1
        if splitted_name[0] == 'Conv':
            n_iter, _ = int(splitted_name[1]), int(splitted_name[2])
            if n_iter > num_iter:
                num_iter = n_iter
    depth_map = [0]*num_iter
    for layer in model.layers:
        name = layer.name
        splitted_name = name.split('_')
        if splitted_name[0] == 'Conv':
            n_iter, ddepth = int(splitted_name[1]), int(splitted_name[2])
            if ddepth > depth_map[n_iter-1]:
                depth_map[n_iter-1] = ddepth
    return num_iter, depth_map, channel_counter

def get_model_dictionary(model):
    model_dictionary = {}
    for layer in model.layers:
        assert layer.name not in model_dictionary
        model_dictionary[layer.name] = layer
    return model_dictionary

# Assumptions:
# - Each iteration generates an embedding, with the exception of the last one
# - Each iteration generates a variance prediction, with the exception of the last one
# - The Second convolutional layer is the one (and only one) that processes the concatenation with the previous embedding
# - Batch Normalization is always before the activation

def decompose_aggregate_model(aggregate_model):
    num_iter, num_conv_per_iter, layer_counter = get_aggregate_model_stats(aggregate_model)
    model_dictionary = get_model_dictionary(aggregate_model)
    
    resulting_models = []
    ####################################################################################################################
    ############################################## First Model Generation ##############################################
    ####################################################################################################################
    cnn_input_shape = model_dictionary['Conv_1_1'].kernel.shape
    input_size = cnn_input_shape[0]*cnn_input_shape[1]*cnn_input_shape[2]

    # Architecture definition
    dense_model_input = Input(input_size, name='patch_input')
    x = Dense(model_dictionary['Conv_1_1'].kernel.shape[-1], activation = model_dictionary['Conv_1_1'].activation)(dense_model_input)
    if 'BN_1_1' in model_dictionary:
        x = BatchNormalization()(x)
    x = model_dictionary['Act_1_1'](x)
    for i in range(1, num_conv_per_iter[0]):
        x = Dense(model_dictionary[f'Conv_1_{str(i+1)}'].kernel.shape[-1], 
                  activation = model_dictionary[f'Conv_1_{str(i+1)}'].activation)(x)
        if f'BN_1_{str(i+1)}' in model_dictionary:
            x = BatchNormalization()(x)
        x = model_dictionary[f'Act_1_{str(i+1)}'](x)
    out_size = model_dictionary['ConvT_1'].kernel.shape[0]*model_dictionary['ConvT_1'].kernel.shape[1]+1
    dense_model_output = Dense(out_size, 
                               activation = model_dictionary['ConvT_1'].activation, name='output_patch')(x)
    dense_model_1 = Model(dense_model_input, {"output": dense_model_output, "embedding_output": x})

    # Weight Adaptation
    dense_model_weights = []
    dense_model_weights.append(np.transpose(model_dictionary['Conv_1_1'].get_weights()[0],(2,3,0,1)).reshape(-1,input_size).T)
    dense_model_weights.append(model_dictionary['Conv_1_1'].get_weights()[1])
    in_ch = dense_model_weights[-2].shape[-1]
    if 'BN_1_1' in model_dictionary:
        dense_model_weights.extend(model_dictionary['BN_1_1'].get_weights())
    
    for i in range(1, num_conv_per_iter[0]):
        dense_model_weights.append(np.transpose(model_dictionary[f'Conv_1_{str(i+1)}'].get_weights()[0],(2,3,0,1)).reshape([-1,in_ch]))
        dense_model_weights.append(model_dictionary[f'Conv_1_{str(i+1)}'].get_weights()[1])
        in_ch = dense_model_weights[-2].shape[-1]
        if f'BN_1_{str(i+1)}' in model_dictionary:
            dense_model_weights.extend(model_dictionary[f'BN_1_{str(i+1)}'].get_weights())

    mu_kernel = np.transpose(model_dictionary['ConvT_1'].get_weights()[0],(2,3,0,1)).reshape(-1,4)
    sigma_kernel = np.transpose(model_dictionary['ConvSig_1'].get_weights()[0],(2,3,0,1)).reshape(-1,1)
    dense_model_weights.append(np.concatenate([mu_kernel, sigma_kernel], axis=-1))
    dense_model_weights.append(dense_model_1.get_weights()[-1])
    dense_model_weights[-1][:4] = model_dictionary['ConvT_1'].get_weights()[1]
    dense_model_weights[-1][-1] = model_dictionary['ConvSig_1'].get_weights()[1]

    dense_model_1.set_weights(dense_model_weights)
    resulting_models.append(dense_model_1)

    ####################################################################################################################
    ######################################### Following Models Generation ##############################################
    ####################################################################################################################
    
    for it in range(1, num_iter):
        it_s = str(it+1)
        embedding_size = resulting_models[-1].output['embedding_output'].shape[-1]
        cnn_input_shape = model_dictionary[f'Conv_{it_s}_1'].kernel.shape
        input_size = cnn_input_shape[0]*cnn_input_shape[1]*cnn_input_shape[2]

        dense_model_input = Input(input_size, name='patch_input')
        embedding_input = Input(embedding_size, name='embedding_input')
        x = Dense(model_dictionary[f'Conv_{it_s}_1'].kernel.shape[-1], 
                  activation = model_dictionary[f'Conv_{it_s}_1'].activation)(dense_model_input)
        if f'BN_{it_s}_1' in model_dictionary:
            x = BatchNormalization()(x)
        x = model_dictionary[f'Act_{it_s}_1'](x)
        x = Concatenate()([embedding_input, x])
        for i in range(1, num_conv_per_iter[it]):
            i_s = str(i+1)
            x = Dense(model_dictionary[f'Conv_{it_s}_{i_s}'].kernel.shape[-1], 
                      activation = model_dictionary[f'Conv_{it_s}_{i_s}'].activation)(x)
            if f'BN_{it_s}_{i_s}' in model_dictionary:
                x = BatchNormalization()(x)
            x = model_dictionary[f'Act_{it_s}_{i_s}'](x)

        out_size = model_dictionary[f'ConvT_{it_s}'].kernel.shape[0]*model_dictionary[f'ConvT_{it_s}'].kernel.shape[1]+int(it < num_iter-1)
        dense_model_output = Dense(out_size, 
                                   activation = model_dictionary[f'ConvT_{it_s}'].activation,
                                   name = 'output_patch')(x)

        if it < num_iter-1:
            dense_model = Model({"input": dense_model_input, "embedding_input": embedding_input}, {"output": dense_model_output, "embedding_output": x})
        else:
            dense_model = Model({"input": dense_model_input, "embedding_input": embedding_input}, dense_model_output)

        # Weight Adaptation
        dense_model_weights = []
        dense_model_weights.append(np.transpose(model_dictionary[f'Conv_{it_s}_1'].get_weights()[0],(2,3,0,1)).reshape(-1,input_size).T)
        dense_model_weights.append(model_dictionary[f'Conv_{it_s}_1'].get_weights()[1])
        in_ch = dense_model_weights[-2].shape[-1]+embedding_size
        if f'BN_{it_s}_1' in model_dictionary:
            dense_model_weights.extend(model_dictionary[f'BN_{it_s}_1'].get_weights())
        
        for i in range(1, num_conv_per_iter[it]):
            i_s = str(i+1)
            dense_model_weights.append(np.transpose(model_dictionary[f'Conv_{it_s}_{i_s}'].get_weights()[0],(2,3,0,1)).reshape([-1,in_ch]))
            if dense_model_weights[-1].shape[1] > dense_model_weights[-1].shape[0]:
                dense_model_weights[-1] = np.transpose(model_dictionary[f'Conv_{it_s}_{i_s}'].get_weights()[0],(2,3,0,1)).reshape([in_ch,-1])
            dense_model_weights.append(model_dictionary[f'Conv_{it_s}_{i_s}'].get_weights()[1])
            in_ch = dense_model_weights[-2].shape[-1]
            if f'BN_{it_s}_{i_s}' in model_dictionary:
                dense_model_weights.extend(model_dictionary[f'BN_{it_s}_{i_s}'].get_weights())

        if it < num_iter-1:
            mu_kernel = np.transpose(model_dictionary[f'ConvT_{it_s}'].get_weights()[0],(2,3,0,1)).reshape(-1,4)
            sigma_kernel = np.transpose(model_dictionary[f'ConvSig_{it_s}'].get_weights()[0],(2,3,0,1)).reshape(-1,1)
            dense_model_weights.append(np.concatenate([mu_kernel, sigma_kernel], axis=-1))
            dense_model_weights.append(dense_model.get_weights()[-1])
            dense_model_weights[-1][:4] = model_dictionary[f'ConvT_{it_s}'].get_weights()[1]
            dense_model_weights[-1][-1] = model_dictionary[f'ConvSig_{it_s}'].get_weights()[1]
        else:
            dense_model_weights.append(np.transpose(model_dictionary[f'ConvT_{it_s}'].get_weights()[0],(2,3,0,1)).reshape(-1,4))
            dense_model_weights.append(dense_model.get_weights()[-1])
            dense_model_weights[-1][:4] = model_dictionary[f'ConvT_{it_s}'].get_weights()[1]

        dense_model.set_weights(dense_model_weights)
        resulting_models.append(dense_model)

    return resulting_models

def iterative2tflite(aggregate_model, input_shape, num_threads):
    models = decompose_aggregate_model(aggregate_model)
    models[2].input_names = ['input', 'embedding']
    iterpreters = [keras2tflite(model, input_shape, num_threads) for model in models]
    return iterpreters