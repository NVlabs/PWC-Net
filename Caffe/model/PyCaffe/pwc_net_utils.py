#!/usr/bin/env python

from __future__ import print_function
import caffe
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import subprocess

# helper function for common structures
def augment_first_image(bottom):    
    img0_aug, img0_aug_params = L.DataAugmentation(bottom, propagate_down=False, ntop=2, 
            augmentation_param=dict(max_multiplier=1, augment_during_test=False, recompute_mean=1000, mean_per_pixel=False, 
            translate   = dict(rand_type="uniform_bernoulli", exp=False, mean= 0, spread= 0.4, prob= 1.0), 
            rotate      = dict(rand_type="uniform_bernoulli", exp=False, mean=0, spread=0.4, prob=1.0), 
            zoom        = dict(rand_type="uniform_bernoulli", exp=True, mean=0.2, spread=0.4, prob=1.0),
            squeeze     = dict(rand_type="uniform_bernoulli", exp=True, mean=0, spread=0.3, prob=1.0 ),
            lmult_pow   = dict(rand_type="uniform_bernoulli", exp=True, mean=-0.2, spread=0.4, prob=1.0 ), 
            lmult_mult  = dict(rand_type="uniform_bernoulli", exp=True, mean=0.0, spread=0.4, prob=1.0 ), 
            lmult_add   = dict(rand_type="uniform_bernoulli", exp=False, mean=0, spread=0.03, prob=1.0 ),
            sat_pow     = dict(rand_type="uniform_bernoulli", exp=True, mean=0, spread=0.4, prob=1.0), 
            sat_mult    = dict(rand_type="uniform_bernoulli", exp=True, mean=-0.3, spread=0.5, prob=1.0), 
            sat_add     = dict(rand_type="uniform_bernoulli", exp=False, mean=0, spread=0.03, prob=1.0), 
            col_pow     = dict(rand_type="gaussian_bernoulli", exp=True, mean=0, spread=0.4, prob=1.0), 
            col_mult    = dict(rand_type="gaussian_bernoulli", exp=True, mean=0, spread=0.2, prob=1.0), 
            col_add     = dict(rand_type="gaussian_bernoulli", exp=False, mean=0, spread=0.02, prob=1.0), 
            ladd_pow    = dict(rand_type="gaussian_bernoulli", exp=True, mean=0, spread=0.4, prob=1.0), 
            ladd_mult   = dict(rand_type="gaussian_bernoulli", exp=True, mean=0.0, spread=0.4, prob=1.0), 
            ladd_add    = dict(rand_type="gaussian_bernoulli", exp=False, mean=0, spread=0.04, prob=1.0), 
            col_rotate  = dict(rand_type="uniform_bernoulli", exp=False, mean=0, spread=1, prob=1.0),
            crop_width=448, crop_height=320, chromatic_eigvec= [0.51, 0.56, 0.65, 0.79, 0.01, -0.62, 0.35, -0.83, 0.44], 
            noise       =dict(rand_type="uniform_bernoulli", exp=False, mean=0.03, spread=0.03, prob=1.0 )
        )   
    )
    return img0_aug, img0_aug_params

def generate_aug_params(img0_aug_params, img0_subtract, img0_aug):
    return L.GenerateAugmentationParameters(img0_aug_params, img0_subtract, img0_aug,
            augmentation_param=dict(augment_during_test=False,
            translate   =dict(rand_type="gaussian_bernoulli", exp= False, mean= 0, spread= 0.03, prob= 1.0), 
            rotate      =dict(rand_type="gaussian_bernoulli", exp=False, mean=0, spread=0.03, prob=1.0), 
            zoom        =dict(rand_type="gaussian_bernoulli", exp=True, mean=0, spread=0.03, prob=1.0),
            gamma       =dict(rand_type="gaussian_bernoulli", exp=True, mean=0, spread=0.02, prob=1.0), 
            brightness  =dict(rand_type="gaussian_bernoulli", exp=False, mean=0, spread=0.02, prob=1.0), 
            contrast    =dict(rand_type="gaussian_bernoulli", exp=True, mean=0, spread=0.02, prob=1.0), 
            color       =dict(rand_type="gaussian_bernoulli", exp=True, mean=0, spread=0.02, prob=1.0)),
            coeff_schedule_param = dict(half_life=50000, initial_coeff=0.5, final_coeff=1)  )

def augment_second_image(img1_subtract, aug_params):
    return L.DataAugmentation(img1_subtract, aug_params, propagate_down=[False, False], 
        augmentation_param=dict(max_multiplier=1, augment_during_test=False, recompute_mean=1000, mean_per_pixel=False,  
        crop_width=448, crop_height=320, chromatic_eigvec= [0.51, 0.56, 0.65, 0.79, 0.01, -0.62, 0.35, -0.83, 0.44]) )

def conv_relu(bottom, num_output, kernel_size=3, pad=1, stride=1):
    layer   = L.Convolution(bottom, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=1, decay_mult=0)],
        convolution_param=dict(num_output=num_output, pad=pad, kernel_size=kernel_size, stride=stride, 
            weight_filler=dict(type='msra'), bias_filler=dict(type='constant'), engine=2 ) ) 
    return L.ReLU(layer, relu_param=dict(negative_slope=0.1), in_place=True)    

def conv(bottom, num_output, kernel_size=3, pad=1, stride=1):
    return L.Convolution(bottom, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=1, decay_mult=0)],
        convolution_param=dict(num_output=num_output, pad=pad, kernel_size=kernel_size, stride=stride, 
            weight_filler=dict(type='msra'), bias_filler=dict(type='constant'), engine=2 ) ) 

def deconv_relu(bottom, num_output, kernel_size=3, pad=1, stride=1):
    layer   = L.Deconvolution(bottom, param=[dict(lr_mult=1, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
        convolution_param=dict(num_output=num_output, pad=pad, kernel_size=kernel_size, stride=stride, 
            weight_filler=dict(type='msra'), bias_filler=dict(type='constant'), engine=2 ) ) 
    return L.ReLU(layer, relu_param=dict(negative_slope=0.1))    

def deconv(bottom, num_output, kernel_size=3, pad=1, stride=1):
    return L.Deconvolution(bottom, param=[dict(lr_mult=1, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
        convolution_param=dict(num_output=num_output, pad=pad, kernel_size=kernel_size, stride=stride, 
            weight_filler=dict(type='msra'), bias_filler=dict(type='constant'), engine=2 ) ) 

def net_conv_relu(net, conv_name, relu_name, bottom, num_output, kernel_size=3, pad=1, stride=1):
    setattr(net, conv_name, L.Convolution(bottom, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=1, decay_mult=0)],
        convolution_param=dict(num_output=num_output, pad=pad, kernel_size=kernel_size, stride=stride, 
            weight_filler=dict(type='msra'), bias_filler=dict(type='constant'), engine=2 ) )  )
    setattr(net, relu_name, L.ReLU(getattr(net, conv_name), relu_param=dict(negative_slope=0.1), in_place=True) )
    # setattr(net, relu_name, L.ReLU(getattr(net, conv_name), relu_param=dict(negative_slope=0.1)) )
    return net, getattr(net, relu_name)

def net_deconv_relu(net, deconv_name, relu_name, bottom, num_output, kernel_size=3, pad=1, stride=1):
    setattr(net, deconv_name, L.Deconvolution(bottom, param=[dict(lr_mult=1, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
        convolution_param=dict(num_output=num_output, pad=pad, kernel_size=kernel_size, stride=stride, 
            weight_filler=dict(type='msra'), bias_filler=dict(type='constant'), engine=2 ) )  )
    setattr(net, relu_name, L.ReLU(getattr(net, deconv_name), relu_param=dict(negative_slope=0.1), in_place=True) )
    return net, getattr(net, relu_name)

def net_conv_relu2(net, input1, input2, num_output, kernel_size=3, pad=1, stride=1, level=1, aux=None):
    output1, output2 =  L.Convolution(input1, input2, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=1, decay_mult=0)],
                        convolution_param=dict(num_output=num_output, pad=pad, kernel_size=kernel_size, stride=stride, 
                        weight_filler=dict(type='msra'), bias_filler=dict(type='constant'), engine=2 ), ntop=2 )
    conv_name0 = 'conv0_{}'.format(level)+aux
    conv_name1 = 'conv1_{}'.format(level)+aux
    relu_name0 = 'Relu0_{}'.format(level)+aux
    relu_name1 = 'Relu1_{}'.format(level)+aux
    setattr(net, conv_name0, output1)
    setattr(net, conv_name1, output2)

    setattr(net, relu_name0, L.ReLU(getattr(net, conv_name0), relu_param=dict(negative_slope=0.1), in_place=True) )
    setattr(net, relu_name1, L.ReLU(getattr(net, conv_name1), relu_param=dict(negative_slope=0.1), in_place=True) )
    return net

def bilinear_interpolation_fixed(net, input, level, num_output=1, pad=1, kernel_size=4, stride=2):

    slice_namex = 'slice_x_{}'.format(level)
    slice_namey = 'slice_y_{}'.format(level)
    slice_conv_namex = 'slice_conv_x_{}'.format(level)
    slice_conv_namey = 'slice_conv_y{}'.format(level)

    input_x, input_y = L.Slice(input, slice_param=dict(axis=1, slice_point=1),  ntop=2)
    setattr(net, slice_namex, input_x)
    setattr(net, slice_namey, input_y)    

    output_x, output_y = L.Deconvolution(input_x, input_y, param=dict(lr_mult=0, decay_mult=0),
        convolution_param=dict(num_output=num_output, pad=pad, kernel_size=kernel_size, stride=stride, 
            weight_filler=dict(type='bilinear'), bias_term=False, engine=2 ),  ntop=2)  
    setattr(net, slice_conv_namex, output_x)
    setattr(net, slice_conv_namey, output_y)
    output = L.Concat(output_x, output_y, concat_param=dict(axis=1))
    return net, output

def bilinear_additive_upsampling(net, input, level, num_output=1, pad=1, kernel_size=4, stride=2):

    # bilinear upsamling
    net, output_b = bilinear_interpolation_fixed(net, input, level, num_output, pad, kernel_size, stride)
    # deconvolutional upsampling
    output_d = deconv(input, 2, 4, 1, 2)
    # residual connection
    output = L.Eltwise(output_b, output_d, eltwise_param=dict(operation=1))   
    return net, output  

#  dilated convolution
def net_dconv_relu(net, conv_name, relu_name, bottom, num_output, kernel_size=3, pad=1, stride=1, dilation=1):
    setattr(net, conv_name, L.Convolution(bottom, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=1, decay_mult=0)],
        convolution_param=dict(num_output=num_output, pad=pad, kernel_size=kernel_size, stride=stride, dilation=dilation,
            weight_filler=dict(type='msra'), bias_filler=dict(type='constant')) )  )
    setattr(net, relu_name, L.ReLU(getattr(net, conv_name), relu_param=dict(negative_slope=0.1), in_place=True) )
    return net, getattr(net, relu_name)


def make_pwc_net_encoder_plus(net, image0, image1):

    # make core net that takes two input image and output predict_flow2
    net = net_conv_relu2(net, image0, image1, 16, 3, 1, 2, level=1, aux = '_a')
    net = net_conv_relu2(net, net.conv0_1_a, net.conv1_1_a, 16, 3, 1, 1, level=1, aux = '_aa')
    net = net_conv_relu2(net, net.conv0_1_aa, net.conv1_1_aa, 16, 3, 1, 1, level=1, aux = '_b')

    net = net_conv_relu2(net, net.conv0_1_b, net.conv1_1_b, 32, 3, 1, 2, level=2, aux = '_a')
    net = net_conv_relu2(net, net.conv0_2_a, net.conv1_2_a, 32, 3, 1, 1, level=2, aux = '_aa')
    net = net_conv_relu2(net, net.conv0_2_aa, net.conv1_2_aa, 32, 3, 1, 1, level=2, aux = '_b')

    net = net_conv_relu2(net, net.conv0_2_b, net.conv1_2_b, 64, 3, 1, 2, level=3, aux = '_a')
    net = net_conv_relu2(net, net.conv0_3_a, net.conv1_3_a, 64, 3, 1, 1, level=3, aux = '_aa')
    net = net_conv_relu2(net, net.conv0_3_aa, net.conv1_3_aa, 64, 3, 1, 1, level=3, aux = '_b')

    net = net_conv_relu2(net, net.conv0_3_b, net.conv1_3_b, 96, 3, 1, 2, level=4, aux = '_a') 
    net = net_conv_relu2(net, net.conv0_4_a, net.conv1_4_a, 96, 3, 1, 1, level=4, aux = '_aa')
    net = net_conv_relu2(net, net.conv0_4_aa, net.conv1_4_aa, 96, 3, 1, 1, level=4, aux = '_b')

    net = net_conv_relu2(net, net.conv0_4_b, net.conv1_4_b, 128, 3, 1, 2, level=5, aux = '_a')
    net = net_conv_relu2(net, net.conv0_5_a, net.conv1_5_a, 128, 3, 1, 1, level=5, aux = '_aa')    
    net = net_conv_relu2(net, net.conv0_5_aa, net.conv1_5_aa, 128, 3, 1, 1, level=5, aux = '_b')    

    net = net_conv_relu2(net, net.conv0_5_b, net.conv1_5_b, 196, 3, 1, 2, level=6, aux = '_a')
    net = net_conv_relu2(net, net.conv0_6_a, net.conv1_6_a, 196, 3, 1, 1, level=6, aux = '_a')
    net = net_conv_relu2(net, net.conv0_6_a, net.conv1_6_a, 196, 3, 1, 1, level=6, aux = '_b')

    net.corr6 = L.Correlation(net.conv0_6_b, net.conv1_6_b, correlation_param=dict(pad=4, kernel_size=1, max_displacement=4, stride_1=1, stride_2=1))
    net.corr6 = L.ReLU(net.corr6, relu_param=dict(negative_slope=0.1), in_place=True)

    # densenet level 6

    # net = densenet(net, level=6)
    net, layer = net_conv_relu(net, 'conv6_0', 'relu6_0', net.corr6, 128, 3, 1, 1) # 128 64 32 16 8 flow?
    net.denseconcat6_0 = layer = L.Concat(net.conv6_0, net.corr6, concat_param=dict(axis=1))
    net, layer = net_conv_relu(net, 'conv6_1', 'relu6_1', layer, 128, 3, 1, 1)  
    net.denseconcat6_1 = layer = L.Concat(net.conv6_1, net.denseconcat6_0, concat_param=dict(axis=1))
    net, layer = net_conv_relu(net, 'conv6_2', 'relu6_2', layer, 96, 3, 1, 1)
    net.denseconcat6_2 = layer = L.Concat(net.conv6_2, net.denseconcat6_1, concat_param=dict(axis=1))
    net, layer = net_conv_relu(net, 'conv6_3', 'relu6_3', layer, 64, 3, 1, 1)
    net.denseconcat6_3 = layer = L.Concat(net.conv6_3, net.denseconcat6_2, concat_param=dict(axis=1))
    net, layer = net_conv_relu(net, 'conv6_4', 'relu6_4', layer, 32, 3, 1, 1)
    net.denseconcat6_4 = layer = L.Concat(net.conv6_4, net.denseconcat6_3, concat_param=dict(axis=1))
    # net.denseconcat6_4 = layer = L.Concat(net.conv6_4, net.conv6_3, net.conv6_2, net.conv6_1, net.conv6_0, concat_param=dict(axis=1))
    net.predict_flow6 = conv(layer, 2, 3, 1, 1)    

    net.upsample_flow_6to5 = deconv(net.predict_flow6, 2, 4, 1, 2)
    # net, net.upsample_flow_6to5 = bilinear_additive_upsampling(net, net.predict_flow6, 4) 

    net.upsample_feature_6to5 = deconv(net.denseconcat6_4, 2, 4, 1, 2) # upsample 2 features

    net.scale_flow_6to5 = L.Eltwise(net.upsample_flow_6to5, eltwise_param=dict(operation=1, coeff=0.625))

    net.warped_image5 = L.Warp(net.conv1_5_b, net.scale_flow_6to5)
    net.corr5 = L.Correlation(net.conv0_5_b, net.warped_image5, correlation_param=dict(pad=4, kernel_size=1, max_displacement=4, stride_1=1, stride_2=1))
    net.corr5 = L.ReLU(net.corr5, relu_param=dict(negative_slope=0.1), in_place=True)
    net.concat5 = L.Concat(net.corr5, net.conv0_5_b, net.upsample_flow_6to5,  net.upsample_feature_6to5, concat_param=dict(axis=1))

    net, layer = net_conv_relu(net, 'conv5_0', 'relu5_0', net.concat5, 128, 3, 1, 1) 
    net.denseconcat5_0 = layer = L.Concat(layer, net.concat5, concat_param=dict(axis=1))
    net, layer = net_conv_relu(net, 'conv5_1', 'relu5_1', layer, 128, 3, 1, 1)  
    net.denseconcat5_1 = layer = L.Concat(layer, net.denseconcat5_0, concat_param=dict(axis=1))
    net, layer = net_conv_relu(net, 'conv5_2', 'relu5_2', layer, 96, 3, 1, 1)
    net.denseconcat5_2 = layer = L.Concat(layer, net.denseconcat5_1, concat_param=dict(axis=1))
    net, layer = net_conv_relu(net, 'conv5_3', 'relu5_3', layer, 64, 3, 1, 1)
    net.denseconcat5_3 = layer = L.Concat(layer, net.denseconcat5_2, concat_param=dict(axis=1))
    net, layer = net_conv_relu(net, 'conv5_4', 'relu5_4', layer, 32, 3, 1, 1)
    net.denseconcat5_4 = layer = L.Concat(layer, net.denseconcat5_3, concat_param=dict(axis=1))

    # net.denseconcat5_4 = layer = L.Concat(net.conv5_4, net.conv5_3, net.conv5_2, net.conv5_1, net.conv5_0, concat_param=dict(axis=1))
    net.predict_flow5 = conv(layer, 2, 3, 1, 1)    


    net.upsample_flow_5to4 = deconv(net.predict_flow5, 2, 4, 1, 2)
    # net, net.upsample_flow_5to4 = bilinear_additive_upsampling(net, net.predict_flow5, 4) 
    net.upsample_feature_5to4 = deconv(net.denseconcat5_4, 2, 4, 1, 2) # upsample 2 features

    net.scale_flow_5to4 = L.Eltwise(net.upsample_flow_5to4, eltwise_param=dict(operation=1, coeff=1.25))
    net.warped_image4 = L.Warp(net.conv1_4_b, net.scale_flow_5to4)
    net.corr4 = L.Correlation(net.conv0_4_b, net.warped_image4, correlation_param=dict(pad=4, kernel_size=1, max_displacement=4, stride_1=1, stride_2=1))
    net.corr4 = L.ReLU(net.corr4, relu_param=dict(negative_slope=0.1), in_place=True)
    net.concat4 = L.Concat(net.corr4, net.conv0_4_b, net.upsample_flow_5to4,  net.upsample_feature_5to4, concat_param=dict(axis=1))

    net, layer = net_conv_relu(net, 'conv4_0', 'relu4_0', net.concat4, 128, 3, 1, 1) # 128 64 32 16 8 flow?
    net.denseconcat4_0 = layer = L.Concat(layer, net.concat4, concat_param=dict(axis=1))
    net, layer = net_conv_relu(net, 'conv4_1', 'relu4_1', layer, 128, 3, 1, 1)  
    net.denseconcat4_1 = layer = L.Concat(layer, net.denseconcat4_0, concat_param=dict(axis=1))
    net, layer = net_conv_relu(net, 'conv4_2', 'relu4_2', layer, 96, 3, 1, 1)
    net.denseconcat4_2 = layer = L.Concat(layer, net.denseconcat4_1, concat_param=dict(axis=1))
    net, layer = net_conv_relu(net, 'conv4_3', 'relu4_3', layer, 64, 3, 1, 1)
    net.denseconcat4_3 = layer = L.Concat(layer, net.denseconcat4_2, concat_param=dict(axis=1))
    net, layer = net_conv_relu(net, 'conv4_4', 'relu4_4', layer, 32, 3, 1, 1)
    net.denseconcat4_4 = layer = L.Concat(layer, net.denseconcat4_3, concat_param=dict(axis=1))
    # net.denseconcat4_4 = layer = L.Concat(net.conv4_4, net.conv4_3, net.conv4_2, net.conv4_1, net.conv4_0, concat_param=dict(axis=1))
    net.predict_flow4 = conv(layer, 2, 3, 1, 1)    


    net.upsample_flow_4to3 = deconv(net.predict_flow4, 2, 4, 1, 2)
    # net, net.upsample_flow_4to3 = bilinear_additive_upsampling(net, net.predict_flow4, 4) 

    net.upsample_feature_4to3 = deconv(net.denseconcat4_4, 2, 4, 1, 2) # upsample 2 features

    net.scale_flow_4to3 = L.Eltwise(net.upsample_flow_4to3, eltwise_param=dict(operation=1, coeff=2.5))

    net.warped_image3 = L.Warp(net.conv1_3_b, net.scale_flow_4to3)
    net.corr3 = L.Correlation(net.conv0_3_b, net.warped_image3, correlation_param=dict(pad=4, kernel_size=1, max_displacement=4, stride_1=1, stride_2=1))
    net.corr3 = L.ReLU(net.corr3, relu_param=dict(negative_slope=0.1), in_place=True)
    net.concat3 = L.Concat(net.corr3, net.conv0_3_b, net.upsample_flow_4to3,  net.upsample_feature_4to3, concat_param=dict(axis=1))

    net, layer = net_conv_relu(net, 'conv3_0', 'relu3_0', net.concat3, 128, 3, 1, 1) # 128 64 32 16 8 flow?
    net.denseconcat3_0 = layer = L.Concat(layer, net.concat3, concat_param=dict(axis=1))
    net, layer = net_conv_relu(net, 'conv3_1', 'relu3_1', layer, 128, 3, 1, 1)  
    net.denseconcat3_1 = layer = L.Concat(layer, net.denseconcat3_0, concat_param=dict(axis=1))
    net, layer = net_conv_relu(net, 'conv3_2', 'relu3_2', layer, 96, 3, 1, 1)
    net.denseconcat3_2 = layer = L.Concat(layer, net.denseconcat3_1, concat_param=dict(axis=1))
    net, layer = net_conv_relu(net, 'conv3_3', 'relu3_3', layer, 64, 3, 1, 1)
    net.denseconcat3_3 = layer = L.Concat(layer, net.denseconcat3_2, concat_param=dict(axis=1))
    net, layer = net_conv_relu(net, 'conv3_4', 'relu3_4', layer, 32, 3, 1, 1)
    net.denseconcat3_4 = layer = L.Concat(layer, net.denseconcat3_3, concat_param=dict(axis=1))
    # net.denseconcat3_4 = layer = L.Concat(net.conv3_4, net.conv3_3, net.conv3_2, net.conv3_1, net.conv3_0, concat_param=dict(axis=1))
    net.predict_flow3 = conv(layer, 2, 3, 1, 1)    



    net.upsample_flow_3to2 = deconv(net.predict_flow3, 2, 4, 1, 2)
    # net, net.upsample_flow_3to2 = bilinear_additive_upsampling(net, net.predict_flow3, 4) 
    net.upsample_feature_3to2 = deconv(net.denseconcat3_4, 2, 4, 1, 2) # upsample 2 features

    net.scale_flow_3to2 = L.Eltwise(net.upsample_flow_3to2, eltwise_param=dict(operation=1, coeff=5.))
    net.warped_image2 = L.Warp(net.conv1_2_b, net.scale_flow_3to2)
    net.corr2 = L.Correlation(net.conv0_2_b, net.warped_image2, correlation_param=dict(pad=4, kernel_size=1, max_displacement=4, stride_1=1, stride_2=1))
    net.corr2 = L.ReLU(net.corr2, relu_param=dict(negative_slope=0.1), in_place=True)
    net.concat2 = L.Concat(net.corr2, net.conv0_2_b, net.upsample_flow_3to2,  net.upsample_feature_3to2, concat_param=dict(axis=1))

    net, layer = net_conv_relu(net, 'conv2_0', 'relu2_0', net.concat2, 128, 3, 1, 1) # 128 64 32 16 8 flow?
    net.denseconcat2_0 = layer = L.Concat(layer, net.concat2, concat_param=dict(axis=1))
    net, layer = net_conv_relu(net, 'conv2_1', 'relu2_1', layer, 128, 3, 1, 1)  
    net.denseconcat2_1 = layer = L.Concat(layer, net.denseconcat2_0, concat_param=dict(axis=1))
    net, layer = net_conv_relu(net, 'conv2_2', 'relu2_2', layer, 96, 3, 1, 1)
    net.denseconcat2_2 = layer = L.Concat(layer, net.denseconcat2_1, concat_param=dict(axis=1))
    net, layer = net_conv_relu(net, 'conv2_3', 'relu2_3', layer, 64, 3, 1, 1)
    net.denseconcat2_3 = layer = L.Concat(layer, net.denseconcat2_2, concat_param=dict(axis=1))
    net, layer = net_conv_relu(net, 'conv2_4', 'relu2_4', layer, 32, 3, 1, 1)
    net.denseconcat2_4 = layer = L.Concat(layer, net.denseconcat2_3, concat_param=dict(axis=1))
    # net.denseconcat2_4 = layer = L.Concat(net.conv2_4, net.conv2_3, net.conv2_2, net.conv2_1, net.conv2_0, concat_param=dict(axis=1))
    
    net.predict_flow_ini = conv(layer, 2, 3, 1, 1)    
    # add dilated convolution as backend
    net, layer = net_conv_relu(net, 'dc_conv1', 'relu_dc1', layer, 128, 3, 1, 1)
    net, layer = net_dconv_relu(net, 'dc_conv2', 'relu_dc2', layer, 128, 3, 2, 1, dilation=2)
    net, layer = net_dconv_relu(net, 'dc_conv3', 'relu_dc3', layer, 128, 3, 4, 1, dilation=4)
    net, layer = net_dconv_relu(net, 'dc_conv4', 'relu_dc4', layer, 96, 3, 8, 1, dilation=8)
    net, layer = net_dconv_relu(net, 'dc_conv5', 'relu_dc5', layer, 64, 3, 16, 1, dilation=16)
    net, layer = net_conv_relu(net, 'dc_conv6', 'relu_dc6', layer, 32, 3, 1, 1)
    net.predict_flow_inc = conv(layer, 2, 3, 1, 1)    

    net.predict_flow2  = L.Eltwise(net.predict_flow_ini, net.predict_flow_inc, eltwise_param=dict(operation=1))  

    return net


def make_net_train(lmdb, preselection, batch_size=8, weights = [0, 0, 0.005, 0.01, 0.02, 0.08, 0.32]):

    net = caffe.NetSpec()

    net.img0, net.img1, net.flow_gt, net.aux= L.CustomData(  
        data_param=dict(source=lmdb, preselection_file = preselection, backend=P.Data.LMDB, batch_size=batch_size, 
            preselection_label=1, rand_permute=True, rand_permute_seed=77, slice_point=[3,6,8], encoding=[1,1,2,3], 
            verbose=True),  ntop=4, include=dict(phase=0))

    net.img0_subtract = L.Eltwise(net.img0, eltwise_param=dict(operation=1,coeff=0.00392156862745))  
    net.img1_subtract = L.Eltwise(net.img1, eltwise_param=dict(operation=1,coeff=0.00392156862745))  

    net.img0_aug, net.img0_aug_params = augment_first_image(net.img0_subtract)

    aug_params      = generate_aug_params(net.img0_aug_params, net.img0_subtract, net.img0_aug)    
    net.img1_aug    = augment_second_image(net.img1_subtract, aug_params)

    net.flow_gt_aug     = L.FlowAugmentation(net.flow_gt, net.img0_aug_params, aug_params, augmentation_param=dict(crop_width=448, crop_height=320))
    net.scaled_flow_gt  = L.Eltwise(net.flow_gt_aug, eltwise_param=dict(operation=1,coeff=0.05))  

    net = make_pwc_net_encoder_plus(net, net.img0_aug, net.img1_aug) 
    
    for i in range(1, len(weights)):
        if weights[i] > 0.:
            scaled_flow_name  = 'scaled_flow_gt{}'.format(i)
            predict_flow_name = 'predict_flow{}'.format(i)
            loss_name         = 'loss{}'.format(i)
            setattr(net, scaled_flow_name, L.Downsample(net.scaled_flow_gt, getattr(net, predict_flow_name), propagate_down=[False, False]) )
            setattr(net, loss_name, L.L1Loss(getattr(net, predict_flow_name), getattr(net, scaled_flow_name), loss_weight=weights[i], l1_loss_param=dict(l2_per_location=True)))
    # loss at level 0: don't scale GT
    if weights[0] > 0.:
        net.loss0 = L.L1Loss(net.predict_flow0, net.scaled_flow_gt, loss_weight=weights[0] , l1_loss_param=dict(l2_per_location=True), propagate_down=[True, False])

    net.Silence0 = L.Silence(net.img0, ntop=0)
    net.Silence1 = L.Silence(net.img1, ntop=0)
    net.Silence2 = L.Silence(net.flow_gt, ntop=0)
    net.Silence3 = L.Silence(net.aux, ntop=0)
    # net.Silence4 = L.Silence(net.predict_flow2_scale, ntop=0)

    return net.to_proto()

def make_net_test(desired_level=2):
    net = caffe.NetSpec()
    net.img0_nomean_resize= L.ImageData(image_data_param=dict(source="tmp/img1.txt",batch_size=1))
    net.img1_nomean_resize= L.ImageData(image_data_param=dict(source="tmp/img2.txt",batch_size=1))
    
    predict_flow_name = 'predict_flow{}'.format(desired_level)

    net = make_pwc_net_encoder_plus(net, net.img0_nomean_resize, net.img1_nomean_resize) 
    net.blob44 = L.Eltwise(getattr(net, predict_flow_name), eltwise_param=dict(operation=1, coeff=20.0))  

    return net.to_proto()
