import torch
import numpy as np

import sys
sys.path.insert(0,'./')
sys.path.insert(0,'./3rd-party/')
from RT4KSR.arch import RT4KSR_Rep
from ECBSR.ecbsr import ECBSR
from ESR.eSR import edgeSR_MAX, edgeSR_TM, edgeSR_TR
from RLFN.rlfn_ntire import RLFN_NTIRE
from RFDN.RFDN import RFDN
from FSRCNN import FSRCNN
from FMEN.train_fmen import TRAIN_FMEN
from ESPCN.model import ESPCN
from AsConvSR.AsConvSR import AsConvSR
from IMDN.imdn import IMDN_RTC, IMDN

from torch.optim.lr_scheduler import StepLR, LambdaLR, ConstantLR
from torch.nn import GELU 

def BPPScheduler(initial_learning_rate, final_learning_rate, starting_epoch, final_epoch, n_iter):
    n_iter += 1
    if n_iter > starting_epoch:
        k = (n_iter-starting_epoch)/(final_epoch-starting_epoch)
        lr = initial_learning_rate - k*(initial_learning_rate-final_learning_rate)
    else:
        lr = initial_learning_rate
    return lr/initial_learning_rate

rt4ksr_config_dictionary = {
    'xxs': [24, 2],
    'xs': [34, 2],
    's': [24,4],
    'm': [34,4],
    'l': [24,6],
    'xl': [32,6],
    'xxl': [24,8],
    'xxxl': [32,8]
}

def get_optimizer(network_type, model, num_epochs):
    if network_type == 'asconvsr':
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.9999))
        scheduler = StepLR(optimizer, step_size=num_epochs//5, gamma=0.5)
    elif network_type.split('_')[0] == 'rt4ksr':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = StepLR(optimizer, step_size=num_epochs//5, gamma=0.5)
    elif network_type == 'fsrcnn':
        optimizer = torch.optim.Adam(
        [
            {"params": model[:-1].parameters(), "lr":1e-3},
            {"params": model[-1:].parameters(), "lr": 1e-4}
        ],
        lr=1e-3,
        )
        scheduler = StepLR(optimizer, step_size=max(num_epochs-20,0), gamma=0.5)
    elif network_type == 'espcn':
        optimizer = torch.optim.Adam(
        [
            {"params": model.conv1.parameters(), "lr":1e-2},
            {"params": model.conv2.parameters(), "lr":1e-2},
            {"params": model.conv3.parameters(), "lr":1e-3}
        ],
        lr=1e-2
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.01**(1/(num_epochs-1)))
    elif network_type.split('_')[0] == 'ecbsr':
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        scheduler = ConstantLR(optimizer, factor=1.0, total_iters=num_epochs)
    elif network_type.split('_')[0][:3] == 'eSR':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = StepLR(optimizer, step_size=int(num_epochs/8.3333), gamma=0.5)   
    elif network_type in ['imdn', 'imdn_rtc']:
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
        scheduler = StepLR(optimizer, step_size=num_epochs//5, gamma=0.5)       
    elif network_type in ['fmen', 'rfdn']:
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        scheduler = StepLR(optimizer, step_size=num_epochs//5, gamma=0.5)         
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = StepLR(optimizer, step_size=max(num_epochs-20,0), gamma=0.1)
    return optimizer, scheduler

def get_training_config(network_type, device):
    config = {}
    num_epochs = 120
    batch_size = 16
    loss_function = torch.nn.MSELoss()
    if network_type.split('_')[0] == 'rt4ksr':
        n_size = network_type.split('_')[-1]
        n_feats, n_blocks = rt4ksr_config_dictionary[n_size]
        config = {'num_channels':1, 
                'num_feats':n_feats, 
                'num_blocks':n_blocks,
                'upscale':2,
                'act':GELU(),
                'eca_gamma':0,
                'is_train':True,
                'forget':False,
                'layernorm':True,
                'residual':False}
        model = RT4KSR_Rep(**config)
        loss_function = torch.nn.L1Loss()
        batch_size = 64
        num_epochs = 140
    elif network_type == 'rfdn':
        num_epochs = 500
        batch_size = 64
        model = RFDN(in_nc=1, nf=50, num_modules=4, out_nc=1, upscale=2)
        loss_function = torch.nn.L1Loss()
    elif network_type == 'rlfn':
        model = RLFN_NTIRE(feature_channels=46, mid_channels=48, upscale=2)
    elif network_type == 'fsrcnn':
        model = FSRCNN()
    elif network_type == 'espcn':
        model = ESPCN()
    elif network_type.split('_')[0] == 'ecbsr':
        batch_size=32
        num_epochs=175
        loss_function = torch.nn.L1Loss()
        n_size = network_type.split('_')[-1]
        if len(n_size) == 0:
            m_nums = 4
            c_nums = 8
        else:
            m_nums = int(n_size.split('C')[0][1:])
            c_nums = int(n_size.split('C')[-1])
        config = {'module_nums':m_nums, 
                'channel_nums':c_nums, 
                'with_idt':False, 
                'act_type':'prelu', 
                'scale':2, 
                'colors':1
        }
        model = ECBSR(**config)
    elif network_type == 'fmen':
        num_epochs = 500
        batch_size = 64
        class Args:
            def __init__(self):
                self.down_blocks = 2
                self.up_blocks = [2, 1, 1, 1, 1]
                self.n_feats = 24
                self.mid_feats = 16
                self.backbone_expand_ratio = 2
                self.attention_expand_ratio = 2

                self.scale = [2]
                self.n_colors = 1
        config = Args()
        loss_function = torch.nn.L1Loss()
        model = TRAIN_FMEN(config)
    elif network_type == 'asconvsr':
        model = AsConvSR(in_ch=1, out_ch=1, scale_factor=2, device=device)
        loss_function = torch.nn.SmoothL1Loss()
        num_epochs = 500
        batch_size = 32
    elif network_type == 'imdn_rtc':
        num_epochs = 500
        batch_size = 16
        loss_function = torch.nn.L1Loss()
        model = IMDN_RTC(in_nc = 1, out_nc=1)
    elif network_type == 'imdn':
        num_epochs = 500
        batch_size = 16
        loss_function = torch.nn.L1Loss()
        model = IMDN(in_nc = 1, out_nc=1, upscale=2)
    elif network_type.split('_')[0] == 'eSR-MAX':
        num_epochs = 250
        batch_size=32
        model = edgeSR_MAX(network_type)
    elif network_type.split('_')[0] == 'eSR-TM':
        model = edgeSR_TM(network_type)
        num_epochs = 250
        batch_size=32
    elif network_type.split('_')[0] == 'eSR-TR':
        model = edgeSR_TR(network_type)
        num_epochs = 250
        batch_size=32
    model = model.to(device)
    optimizer, scheduler = get_optimizer(network_type, model, num_epochs)
    return model, config, loss_function, optimizer, num_epochs, batch_size, scheduler