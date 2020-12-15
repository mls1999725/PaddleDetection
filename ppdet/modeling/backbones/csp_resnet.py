# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant
from collections import OrderedDict
from ppdet.core.workspace import register, serializable

__all__ = [                                                                                                                                                                                            
    "CSPResNet50vd"
]
'''
__all__ = [
    "CSPResNet50_leaky", "CSPResNet50_mish", "CSPResNet101_leaky",
    "CSPResNet101_mish"
]
'''

@register
@serializable
class CSPResNet50vd(object):
    def __init__(self, layers=50, act="leaky_relu", feature_maps=[2, 3, 4, 5], dcn_v2_stages=[], weight_prefix_name=''):
        super(CSPResNet50vd, self).__init__()
        self.layers = layers
        self.act = act
        self.dcn_v2_stages = dcn_v2_stages
        self.prefix_name = weight_prefix_name
        self.feature_maps = feature_maps

    def __call__(self, input):
        layers = self.layers
        supported_layers = [50, 101]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(
                supported_layers, layers)
        
        if layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]

        num_filters = [64, 128, 256, 512]
        data_format = "NCHW"

        conv = self.conv_bn_layer(
            input=input,
            num_filters=64,
            filter_size=7,
            stride=2,
            act=self.act,
            name="conv1",
            data_format=data_format)
        conv = fluid.layers.pool2d(
            input=conv,
            pool_size=2,
            pool_stride=2,
            pool_padding=0,
            pool_type='max',
            data_format=data_format)

        res_endpoints = []

        for block in range(len(depth)):
            conv_name = "res" + str(block + 2) + chr(97)
            if block != 0:
                conv = self.conv_bn_layer(
                    input=conv,
                    num_filters=num_filters[block],
                    filter_size=3,
                    stride=2,
                    act=self.act,
                    name=conv_name + "_downsample",
                    data_format=data_format)

            # split
            left = conv
            right = conv
            if block == 0:
                ch = num_filters[block]
            else:
                ch = num_filters[block] * 2
            right = self.conv_bn_layer(
                input=right,
                num_filters=ch,
                filter_size=1,
                act=self.act,
                name=conv_name + "_right_first_route",
                data_format=data_format)

            dcn_v2 = True if (block+2) in self.dcn_v2_stages else False
 
            for i in range(depth[block]):
                conv_name = "res" + str(block + 2) + chr(97 + i)
                is_first = False if ((block+2) != 2) else True

                right = self.bottleneck_block(
                    input=right,
                    num_filters=num_filters[block],
                    stride=1,
                    name=conv_name,
                    is_first=is_first,
                    data_format=data_format,
                    dcn_v2=dcn_v2)

            # route
            left = self.conv_bn_layer(
                input=left,
                num_filters=num_filters[block] * 2,
                filter_size=1,
                act=self.act,
                name=conv_name + "_left_route",
                data_format=data_format)
            right = self.conv_bn_layer(
                input=right,
                num_filters=num_filters[block] * 2,
                filter_size=1,
                act=self.act,
                name=conv_name + "_right_route",
                data_format=data_format)
            conv = fluid.layers.concat([left, right], axis=1)

            conv = self.conv_bn_layer(
                input=conv,
                num_filters=num_filters[block] * 2,
                filter_size=1,
                stride=1,
                act=self.act,
                name=conv_name + "_merged_transition",
                data_format=data_format)

            if (block+2) in self.feature_maps:
                res_endpoints.append(conv)

        return OrderedDict([('res{}_sum'.format(self.feature_maps[idx]), feat)
                                for idx, feat in enumerate(res_endpoints)])

    def conv_bn_layer(self,
                      input,
                      num_filters,
                      filter_size,
                      stride=1,
                      groups=1,
                      act=None,
                      name=None,
                      data_format='NCHW',
                      dcn_v2=False):
        _name = self.prefix_name + name if self.prefix_name != '' else name

        if not dcn_v2:
            conv = fluid.layers.conv2d(
                input=input,
                num_filters=num_filters,
                filter_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                groups=groups,
                act=None,
                param_attr=ParamAttr(name=_name + "_weights"),
                bias_attr=False,
                name=name + '.conv2d.output.1',
                data_format=data_format)
        else:
            # select deformable conv"
            offset_mask = self._conv_offset(
                input=input,
                filter_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                act=None,
                name=_name + "_conv_offset")
            offset_channel = filter_size**2 * 2
            mask_channel = filter_size**2
            offset, mask = fluid.layers.split(
                input=offset_mask,
                num_or_sections=[offset_channel, mask_channel],
                dim=1)
            mask = fluid.layers.sigmoid(mask)
            conv = fluid.layers.deformable_conv(
                input=input,
                offset=offset,
                mask=mask,
                num_filters=num_filters,
                filter_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                groups=groups,
                deformable_groups=1,
                im2col_step=1,
                param_attr=ParamAttr(
                    name=_name + "_weights"),
                bias_attr=False,
                name=_name + ".conv2d.output.1")

        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        bn = fluid.layers.batch_norm(
            input=conv,
            act=None,
            name=bn_name + '.output.1',
            param_attr=ParamAttr(name=bn_name + '_scale'),
            bias_attr=ParamAttr(bn_name + '_offset'),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance',
            data_layout=data_format)
        if act == "relu":
            bn = fluid.layers.relu(bn)
        elif act == "leaky_relu":
            bn = fluid.layers.leaky_relu(bn)
        elif act == "mish":
            bn = self._mish(bn)
        return bn

    def _conv_offset(self,
                     input,
                     filter_size,
                     stride,
                     padding,
                     act=None,
                     name=None):
        out_channel = filter_size * filter_size * 3
        out = fluid.layers.conv2d(
            input,
            num_filters=out_channel,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            param_attr=ParamAttr(
                initializer=Constant(0.0), name=name + ".w_0"),
            bias_attr=ParamAttr(
                initializer=Constant(0.0), name=name + ".b_0"),
            act=act,
            name=name)
        return out

    def _mish(self, input):
        return input * fluid.layers.tanh(self._softplus(input))

    def _softplus(self, input):
        expf = fluid.layers.exp(fluid.layers.clip(input, -200, 50))
        return fluid.layers.log(1 + expf)

    def shortcut(self, input, ch_out, stride, is_first, name, data_format):
        max_pooling_in_short_cut = True
        ch_in = input.shape[1]

        if ch_in != ch_out or stride != 1 or (self.layers < 50 and is_first):
            if max_pooling_in_short_cut and not is_first:
                input = fluid.layers.pool2d(
                    input=input,
                    pool_size=2,
                    pool_stride=2,
                    pool_padding=0,
                    ceil_mode=True,
                    pool_type='avg')
                return self.conv_bn_layer(
                input, ch_out, 1, 1, name=name, data_format=data_format)
            return self.conv_bn_layer(
                input, ch_out, 1, stride, name=name, data_format=data_format)
        else:
            return input

    def bottleneck_block(self, input, num_filters, stride, name, is_first, data_format, dcn_v2=False):
        conv0 = self.conv_bn_layer(
            input=input,
            num_filters=num_filters,
            filter_size=1,
            act="leaky_relu",
            name=name + "_branch2a",
            data_format=data_format)
        conv1 = self.conv_bn_layer(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act="leaky_relu",
            name=name + "_branch2b",
            data_format=data_format,
            dcn_v2=dcn_v2)
        conv2 = self.conv_bn_layer(
            input=conv1,
            num_filters=num_filters * 2,
            filter_size=1,
            act=None,
            name=name + "_branch2c",
            data_format=data_format)

        short = self.shortcut(
            input,
            num_filters * 2,
            stride,
            is_first=is_first,
            name=name + "_branch1",
            data_format=data_format)

        ret = short + conv2
        ret = fluid.layers.leaky_relu(ret, alpha=0.1)
        return ret

'''
def CSPResNet50_leaky():
    model = CSPResNet(layers=50, act="leaky_relu")
    return model


def CSPResNet50_mish():
    model = CSPResNet(layers=50, act="mish")
    return model


def CSPResNet101_leaky():
    model = CSPResNet(layers=101, act="leaky_relu")
    return model


def CSPResNet101_mish():
    model = CSPResNet(layers=101, act="mish")
    return model
'''
