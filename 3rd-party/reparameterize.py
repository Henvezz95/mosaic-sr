import torch
import torch.nn.functional as F

def merge_bn(w, b, gamma, beta, mean, var, eps, before_conv=True):
    """Merge BN layer into convolution layer.

    Args:
        w (torch.tensor): Convolution kernel weight. (C_out, C_in, K, K)
        b (torch.tensor): Convolution kernel bias. (C_out)
    """

    out_feats = w.shape[0]
    std = (var + eps).sqrt()
    scale = gamma / std
    bn_bias = beta - mean * gamma / std

    # Reparameterizing kernel
    if before_conv:
        rep_w = w * scale.reshape(1, -1, 1, 1)
    else:
        rep_w = torch.mm(torch.diag(scale), w.view(out_feats, -1)).view(w.shape)

    # Reparameterizing bias
    if before_conv:
        rep_b = torch.mm(torch.sum(w, dim=(2,3)), bn_bias.unsqueeze(1)).squeeze() + b
    else:
        rep_b = b.mul(scale) + bn_bias
        
    return rep_w, rep_b


def bn_parameter(pretrain_state_dict, k, dst='bn1'):
    src = k.split('.')[-2]
    gamma = pretrain_state_dict[k.replace(src, dst)]
    beta = pretrain_state_dict[k.replace(f'{src}.weight', f'{dst}.bias')]
    mean = pretrain_state_dict[k.replace(f'{src}.weight', f'{dst}.running_mean')]
    var = pretrain_state_dict[k.replace(f'{src}.weight', f'{dst}.running_var')]
    eps = 1e-05

    return gamma, beta, mean, var, eps


def reparameterize(net, rep_model, device='cuda'):
    rep_state_dict = rep_model.state_dict()
    pretrained_state_dict = net.state_dict()
    
    for k, v in rep_state_dict.items():            
        if "rep_conv.weight" in k:
            # merge conv1x1-conv3x3-conv1x1
            k0 = pretrained_state_dict[k.replace("rep", "expand")]
            k1 = pretrained_state_dict[k.replace("rep", "fea")]
            k2 = pretrained_state_dict[k.replace("rep", "reduce")]
            
            bias_str = k.replace("weight", "bias")
            b0 = pretrained_state_dict[bias_str.replace("rep", "expand")]
            b1 = pretrained_state_dict[bias_str.replace("rep", "fea")]
            b2 = pretrained_state_dict[bias_str.replace("rep", "reduce")]
            
            mid_feats, n_feats = k0.shape[:2]

            # first step: remove the middle identity
            for i in range(mid_feats):
                k1[i, i, 1, 1] += 1.0
        
            # second step: merge the first 1x1 convolution and the next 3x3 convolution
            merged_k0k1 = F.conv2d(input=k1, weight=k0.permute(1, 0, 2, 3))
            merged_b0b1 = b0.view(1, -1, 1, 1) * torch.ones(1, mid_feats, 3, 3).to(device)
            merged_b0b1 = F.conv2d(input=merged_b0b1, weight=k1, bias=b1)

            # third step: merge the remain 1x1 convolution
            merged_k0k1k2 = F.conv2d(input=merged_k0k1.permute(1, 0, 2, 3), weight=k2).permute(1, 0, 2, 3)
            merged_b0b1b2 = F.conv2d(input=merged_b0b1, weight=k2, bias=b2).view(-1)

            # last step: remove the global identity
            for i in range(n_feats):
                merged_k0k1k2[i, i, 1, 1] += 1.0
            
            # save merged weights and biases in rep state dict
            rep_state_dict[k] = merged_k0k1k2.float()
            rep_state_dict[bias_str] = merged_b0b1b2.float()
            
        elif "rep_conv.bias" in k:
            pass
            
        # merge BN
        elif 'squeeze.weight' in k:
            bias_str = k.replace('weight', 'bias')
            w = pretrained_state_dict[k]
            b = pretrained_state_dict[bias_str]
            gamma, beta, mean, var, eps = bn_parameter(pretrained_state_dict, k, dst='bn1')

            rep_w, rep_b = merge_bn(w, b, gamma, beta, mean, var, eps, before_conv=True)

            rep_state_dict[k] = rep_w
            rep_state_dict[bias_str] = rep_b

        elif 'squeeze.bias' in k:
            pass

        elif 'excitate.weight' in k:
            bias_str = k.replace('weight', 'bias')
            w = pretrained_state_dict[k]
            b = pretrained_state_dict[bias_str]
            gamma1, beta1, mean1, var1, eps1 = bn_parameter(pretrained_state_dict, k, dst='bn2')
            gamma2, beta2, mean2, var2, eps2 = bn_parameter(pretrained_state_dict, k, dst='bn3')
            rep_w, rep_b = merge_bn(w, b, gamma1, beta1, mean1, var1, eps1, before_conv=True)
            rep_w, rep_b = merge_bn(rep_w, rep_b, gamma2, beta2, mean2, var2, eps2, before_conv=False)

            rep_state_dict[k] = rep_w
            rep_state_dict[bias_str] = rep_b

        elif 'excitate.bias' in k:
            pass

        elif k in pretrained_state_dict.keys():
            rep_state_dict[k] = pretrained_state_dict[k]

        else:
            raise NotImplementedError('{} is not found in pretrain_state_dict.'.format(k))

    rep_model.load_state_dict(rep_state_dict, strict=True)
    return rep_model

def reparameterize_ECBSR(model_ecbsr , model_plain):
    ## copy weights from ecbsr to plainsr
    depth = len(model_ecbsr.backbone)
    for d in range(depth):
        module = model_ecbsr.backbone[d]
        act_type = module.act_type
        RK, RB = module.rep_params()
        model_plain.backbone[d].conv3x3.weight.data = RK
        model_plain.backbone[d].conv3x3.bias.data = RB

        if act_type == 'relu':     pass
        elif act_type == 'linear': pass
        elif act_type == 'prelu':  model_plain.backbone[d].act.weight.data = module.act.weight.data
        else: raise ValueError('invalid type of activation!')
    return model_plain