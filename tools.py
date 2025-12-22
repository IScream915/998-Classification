import torch
import torch.nn as nn

from infotool.helper import clever_format

import copy

def profile_origin(model, inputs):
    """Simple profile function to calculate FLOPs and parameters"""
    total_flops = 0
    total_params = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Calculate FLOPs for convolution
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_h, kernel_w = module.kernel_size
            H_in, W_in = inputs[0].shape[2], inputs[0].shape[3]

            # Standard convolution FLOPs calculation
            flops = H_in * W_in * in_channels * kernel_h * kernel_w * out_channels
            if module.bias is not None:
                flops += H_in * W_in * out_channels

            total_flops += flops

            # Calculate parameters
            params = in_channels * kernel_h * kernel_w * out_channels
            if module.bias is not None:
                params += out_channels
            total_params += params

        elif isinstance(module, nn.Linear):
            # Calculate FLOPs for fully connected layer
            in_features = module.in_features
            out_features = module.out_features
            flops = in_features * out_features
            if module.bias is not None:
                flops += out_features

            total_flops += flops

            # Calculate parameters
            params = in_features * out_features
            if module.bias is not None:
                params += out_features
            total_params += params

    return total_flops, total_params

def convert_syncbn_to_bn(module):
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm.SyncBatchNorm):
        module_output = torch.nn.BatchNorm2d(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
        )
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig

    for name, child in module.named_children():
        module_output.add_module(
            name, convert_syncbn_to_bn(child)
        )
    del module
    return module_output


def cal_flops_params(original_model, input_size):
    model = copy.deepcopy(original_model)
    model = convert_syncbn_to_bn(model)
    input_size = list(input_size)
    assert len(input_size) in [3, 4]
    if len(input_size) == 4:
        if input_size[0] != 1:
            print('modify batchsize of input_size from {} to 1'.format(input_size[0]))
            input_size[0] = 1

    if len(input_size) == 3:
        input_size.insert(0, 1)

    flops, params = profile_origin(model, inputs=(torch.zeros(input_size), ))

    print('flops = {}, params = {}'.format(flops, params))
    print('flops = {}, params = {}'.format(clever_format(flops), clever_format(params)))

    return flops, params
