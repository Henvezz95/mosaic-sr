import torch
from torch.nn import Sequential, Conv2d, LeakyReLU, ConvTranspose2d, GELU, PReLU
from collections import OrderedDict
import torch.nn.functional as F

def FSRCNN(expand_init = 56, init_feat = 16, mid_feat = 12, expand_end=56):
    layer_dictionary = OrderedDict()
    layer_dictionary['conv_1'] = Conv2d(1, expand_init, kernel_size=(5,5), padding='same')
    layer_dictionary['act_1'] = PReLU(expand_init)
    layer_dictionary['conv_2'] = Conv2d(expand_init, init_feat, kernel_size=(1,1), padding='same')
    layer_dictionary['act_2'] = PReLU(init_feat)
    layer_dictionary['conv_3'] = Conv2d(init_feat, mid_feat, kernel_size=(3,3), padding='same')
    layer_dictionary['act_3'] = PReLU(mid_feat)
    layer_dictionary['conv_4'] = Conv2d(mid_feat, mid_feat, kernel_size=(3,3), padding='same')
    layer_dictionary['act_4'] = PReLU(mid_feat)
    layer_dictionary['conv_5'] = Conv2d(mid_feat, mid_feat, kernel_size=(3,3), padding='same')
    layer_dictionary['act_5'] = PReLU(mid_feat)
    layer_dictionary['conv_6'] = Conv2d(mid_feat, mid_feat, kernel_size=(3,3), padding='same')
    layer_dictionary['act_6'] = PReLU(mid_feat)
    layer_dictionary['conv_7'] = Conv2d(mid_feat, expand_end, kernel_size=(1,1), padding='same')
    layer_dictionary['act_7'] = PReLU(expand_end)
    layer_dictionary['upscale'] = ConvTranspose2d(expand_end, 1, (9, 9), (2, 2), (4, 4), (1, 1))
    model = Sequential(layer_dictionary)
    return model