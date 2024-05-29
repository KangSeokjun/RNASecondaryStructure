import re
from itertools import chain
from math import ceil

from typing import Optional, Sequence, Union

import torch
from torch.nn import Module, Conv2d, BatchNorm2d, Linear, init, Sequential, ReLU, Dropout2d, MaxPool2d, Upsample
from torch.nn import Sequential, ReLU
from torch.nn import functional as F

class RichRepr(object):
    """
    Allows to modify the normal __repr__ output of a torch.nn.Module,
    adding info as positional and keyword arguments
    """
    def __repr__(self, *args, **kwargs):
        res = super(RichRepr, self).__repr__()
        args = filter(lambda s: len(s) > 0, map(str, args))
        kwargs = (f'{k}={v}' for k, v in kwargs.items())
        desc = ', '.join(chain(args, kwargs))
        return re.sub(rf'({self.__class__.__name__})', rf'\1({desc})', res, count=1)
      
class Concatenate(Module):
    def forward(self, x, y):
        res = torch.cat([x, y], dim=1)
        return res

class BasicConv2d(Module):
  def __init__(self, input_channels, output_channels, **kwargs):
    super().__init__()
    self.conv= Conv2d(input_channels, output_channels, bias=False, **kwargs)
    self.bn= BatchNorm2d(output_channels)
    self.relu= ReLU(inplace=True)

  def forward(self, x):
    x= self.conv(x)
    x= self.bn(x)
    x= self.relu(x)

    return x

class Bottleneck(RichRepr, Sequential):
    r"""
    A 1x1 convolutional layer, followed by Batch Normalization and ReLU
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(Bottleneck, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.add_module('conv', Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
        self.add_module('norm', BatchNorm2d(num_features=out_channels))
        self.add_module('relu', ReLU(inplace=True))

    def __repr__(self):
        return super(Bottleneck, self).__repr__(self.in_channels, self.out_channels)

class DenseLayer(RichRepr, Sequential):
    r"""
    Dense Layer as described in [DenseNet](https://arxiv.org/abs/1608.06993)
    and implemented in https://github.com/liuzhuang13/DenseNet

    Consists of:

    - Batch Normalization
    - ReLU
    - (Bottleneck)
    - 3x3 Convolution
    - (Dropout)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 bottleneck_ratio: Optional[int] = None, dropout: float = 0.0):
        super(DenseLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.add_module('norm', BatchNorm2d(num_features=in_channels))
        self.add_module('relu', ReLU(inplace=True))

        if bottleneck_ratio is not None:
            self.add_module('bottleneck', Bottleneck(in_channels, bottleneck_ratio * out_channels))
            in_channels = bottleneck_ratio * out_channels

        self.add_module('conv', Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))

        if dropout > 0:
            self.add_module('drop', Dropout2d(dropout, inplace=True))

    def __repr__(self):
        return super(DenseLayer, self).__repr__(self.in_channels, self.out_channels)
  
class TransitionDown(RichRepr, Sequential):
    r"""
    Transition Down Block as described in [FCDenseNet](https://arxiv.org/abs/1611.09326),
    plus compression from [DenseNet](https://arxiv.org/abs/1608.06993)

    Consists of:
    - Batch Normalization
    - ReLU
    - 1x1 Convolution (with optional compression of the number of channels)
    - (Dropout)
    - 2x2 Max Pooling
    """

    def __init__(self, in_channels: int, compression: float = 1.0, dropout: float = 0.0):
        super(TransitionDown, self).__init__()

        if not 0.0 < compression <= 1.0:
            raise ValueError(f'Compression must be in (0, 1] range, got {compression}')

        self.in_channels = in_channels
        self.dropout = dropout
        self.compression = compression
        self.out_channels = int(ceil(compression * in_channels))

        self.add_module('norm', BatchNorm2d(num_features=in_channels))
        self.add_module('relu', ReLU(inplace=True))
        self.add_module('conv', Conv2d(in_channels, self.out_channels, kernel_size=1, bias=False))

        if dropout > 0:
            self.add_module('drop', Dropout2d(dropout))

        self.add_module('pool', MaxPool2d(kernel_size=2, stride=2))

    def __repr__(self):
        return super(TransitionDown, self).__repr__(self.in_channels, self.out_channels, dropout=self.dropout)
      
class TransitionUp(RichRepr, Module):
    r"""
    Transition Up Block as described in [FCDenseNet](https://arxiv.org/abs/1611.09326)

    The block upsamples the feature map and concatenates it with the feature map coming from the skip connection.
    If the two maps don't overlap perfectly they are first aligened centrally and cropped to match.
    """

    def __init__(self, upsample_channels: int, skip_channels: Optional[int] = None):
        r"""
        :param upsample_channels: number of channels from the upsampling path
        :param skip_channels: number of channels from the skip connection, it is not required,
                              but if specified allows to statically compute the number of output channels
        """
        super(TransitionUp, self).__init__()

        self.up_channels= int(upsample_channels/2)
        self.skip_channels = skip_channels
        self.out_channels= self.up_channels

#        self.add_module('upconv', ConvTranspose2d(self.upsample_channels, self.upsample_channels,kernel_size=3, stride=2, padding=0, bias=True))
#        self.add_module('concat', CenterCropConcat())

        self.add_module('upconv', Sequential(Upsample(scale_factor=2),
                        Conv2d(upsample_channels,self.up_channels,kernel_size=3,stride=1,padding=1,bias=True),
                        BatchNorm2d(self.up_channels),
                        ReLU(inplace=True)))
        self.add_module('concat', Concatenate())


    def forward(self, upsample, skip):
        if self.skip_channels is not None and skip.shape[1] != self.skip_channels:
            raise ValueError(f'Number of channels in the skip connection input ({skip.shape[1]}) '
                             f'is different from the expected number of channels ({self.skip_channels})')
        res = self.upconv(upsample)
        res= res+skip
        return res

    def __repr__(self):
        skip_channels = self.skip_channels if self.skip_channels is not None else "?"
        out_channels = self.out_channels if self.out_channels is not None else "?"
        return super(TransitionUp, self).__repr__(f'[{self.up_channels}, {skip_channels}] -> {out_channels})')

class DenseBlock(RichRepr, Module):
    r"""
    Dense Block as described in [DenseNet](https://arxiv.org/abs/1608.06993)
    and implemented in https://github.com/liuzhuang13/DenseNet

    - Consists of several DenseLayer (possibly using a Bottleneck and Dropout) with the same output shape
    - The first DenseLayer is fed with the block input
    - Each subsequent DenseLayer is fed with a tensor obtained by concatenating the input and the output
      of the previous DenseLayer on the channel axis
    - The block output is the concatenation of the output of every DenseLayer, and optionally the block input,
      so it will have a channel depth of (growth_rate * num_layers) or (growth_rate * num_layers + in_channels)
    """

    def __init__(self, in_channels: int, growth_rate: int, num_layers: int,
                 concat_input: bool = False, dense_layer_params: Optional[dict] = None):
        super(DenseBlock, self).__init__()

        self.concat_input = concat_input
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.num_layers = num_layers
        self.out_channels = growth_rate * num_layers
        if self.concat_input:
            self.out_channels += self.in_channels

        if dense_layer_params is None:
            dense_layer_params = {}

        for i in range(num_layers):
            self.add_module(
                f'layer_{i}',
                DenseLayer(in_channels=in_channels + i * growth_rate, out_channels=growth_rate, **dense_layer_params)
            )

    def forward(self, block_input):
        layer_input = block_input
        # empty tensor (not initialized) + shape=(0,)
        layer_output = block_input.new_empty(0)

        all_outputs = [block_input] if self.concat_input else []
        for layer in self._modules.values():
            layer_input = torch.cat([layer_input, layer_output], dim=1)
            layer_output = layer(layer_input)
            all_outputs.append(layer_output)

        return torch.cat(all_outputs, dim=1)

    def __repr__(self):
        concat_input = f'+{self.in_channels}' if self.concat_input else ''
        out_channels = f'{self.num_layers}*{self.growth_rate}{concat_input}={self.out_channels}'
        return super(DenseBlock, self).__repr__(self.in_channels, out_channels)

class FCDenseNet(Module):
    r"""
    The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation
    https://arxiv.org/abs/1611.09326

    In this paper, we extend DenseNets to deal with the problem of semantic segmentation. We achieve state-of-the-art
    results on urban scene benchmark datasets such as CamVid and Gatech, without any further post-processing module nor
    pretraining. Moreover, due to smart construction of the model, our approach has much less parameters than currently
    published best entries for these datasets.
    """

    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 1000,
                 initial_num_features: int = 48,
                 dropout: float = 0.0,

                 down_dense_growth_rates: Union[int, Sequence[int]] = 16,
                 down_dense_bottleneck_ratios: Union[Optional[int], Sequence[Optional[int]]] = None,
                 down_dense_num_layers: Union[int, Sequence[int]] = (4, 5, 7, 10, 12),
                 down_transition_compression_factors: Union[float, Sequence[float]] = 1.0,

                 middle_dense_growth_rate: int = 16,
                 middle_dense_bottleneck: Optional[int] = None,
                 middle_dense_num_layers: int = 15,

                 up_dense_growth_rates: Union[int, Sequence[int]] = 16,
                 up_dense_bottleneck_ratios: Union[Optional[int], Sequence[Optional[int]]] = None,
                 up_dense_num_layers: Union[int, Sequence[int]] = (12, 10, 7, 5, 4)):
        super(FCDenseNet, self).__init__()

        # region Parameters handling
        self.in_channels = in_channels
        self.out_channels = out_channels

        if type(down_dense_growth_rates) == int:
            down_dense_growth_rates = (down_dense_growth_rates,) * 5
        if down_dense_bottleneck_ratios is None or type(down_dense_bottleneck_ratios) == int:
            down_dense_bottleneck_ratios = (down_dense_bottleneck_ratios,) * 5
        if type(down_dense_num_layers) == int:
            down_dense_num_layers = (down_dense_num_layers,) * 5
        if type(down_transition_compression_factors) == float:
            down_transition_compression_factors = (down_transition_compression_factors,) * 5

        if type(up_dense_growth_rates) == int:
            up_dense_growth_rates = (up_dense_growth_rates,) * 5
        if up_dense_bottleneck_ratios is None or type(up_dense_bottleneck_ratios) == int:
            up_dense_bottleneck_ratios = (up_dense_bottleneck_ratios,) * 5
        if type(up_dense_num_layers) == int:
            up_dense_num_layers = (up_dense_num_layers,) * 5
        # endregion

        # region First convolution
        # The Lasagne implementation uses convolution with 'same' padding, the PyTorch equivalent is padding=1
        self.features= Sequential(BasicConv2d(in_channels, 64, kernel_size=3,padding=1),
                                  BasicConv2d(64, 32, kernel_size=3,padding=1),
                                  BasicConv2d(32, initial_num_features, kernel_size=3,padding=1),)

        current_channels = initial_num_features
        # endregion

        # region Downward path
        # Pairs of Dense Blocks with input concatenation and TransitionDown layers
        down_dense_params = [
            {
                'concat_input': True,
                'growth_rate': gr,
                'num_layers': nl,
                'dense_layer_params': {
                    'dropout': dropout,
                    'bottleneck_ratio': br
                }
            }
            for gr, nl, br in
            zip(down_dense_growth_rates, down_dense_num_layers, down_dense_bottleneck_ratios)
        ]
        down_transition_params = [
            {
                'dropout': dropout,
                'compression': c
            } for c in down_transition_compression_factors
        ]
        skip_connections_channels = []

        self.down_dense = Module()
        self.down_trans = Module()
        down_pairs_params = zip(down_dense_params, down_transition_params)
        for i, (dense_params, transition_params) in enumerate(down_pairs_params):
            block = DenseBlock(current_channels, **dense_params)
            current_channels = block.out_channels
            self.down_dense.add_module(f'block_{i}', block)

            skip_connections_channels.append(block.out_channels)

            transition = TransitionDown(current_channels, **transition_params)
            current_channels = transition.out_channels
            self.down_trans.add_module(f'trans_{i}', transition)
        # endregion

        # region Middle block
        # Renamed from "bottleneck" in the paper, to avoid confusion with the Bottleneck of DenseLayers
        self.middle = DenseBlock(
            current_channels,
            middle_dense_growth_rate,
            middle_dense_num_layers,
            concat_input=True,
            dense_layer_params={
                'dropout': dropout,
                'bottleneck_ratio': middle_dense_bottleneck
            })
        current_channels = self.middle.out_channels

        # endregion

        # region Upward path
        # Pairs of TransitionUp layers and Dense Blocks without input concatenation
        up_transition_params = [
            {
                'skip_channels': sc,
            } for sc in reversed(skip_connections_channels)
        ]
        up_dense_params = [
            {
                'concat_input': False,
                'growth_rate': gr,
                'num_layers': nl,
                'dense_layer_params': {
                    'dropout': dropout,
                    'bottleneck_ratio': br
                }
            }
            for gr, nl, br in
            zip(up_dense_growth_rates, up_dense_num_layers, up_dense_bottleneck_ratios)
        ]

        self.up_dense = Module()
        self.up_trans = Module()
        up_pairs_params = zip(up_transition_params, up_dense_params)
        for i, (transition_params_up, dense_params_up) in enumerate(up_pairs_params):
            transition = TransitionUp(current_channels, **transition_params_up)
            current_channels = transition.out_channels
            self.up_trans.add_module(f'trans_{i}', transition)

            block = DenseBlock(current_channels, **dense_params_up)
            current_channels = block.out_channels
            self.up_dense.add_module(f'block_{i}', block)
        # endregion



        # region Final convolution
        self.final= Conv2d(current_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.Norm= BatchNorm2d(num_features= 1)

        
        # endregion

        # region Weight initialization
        for module in self.modules():
            if isinstance(module, Conv2d):
                init.kaiming_normal_(module.weight)
            elif isinstance(module, BatchNorm2d):
                module.reset_parameters()
            elif isinstance(module, Linear):
                init.xavier_uniform_(module.weight)
                init.constant_(module.bias, 0)
        # endregion

    def forward(self, x):
    
        res = self.features(x)


        skip_tensors = []
        for dense, trans in zip(self.down_dense.children(), self.down_trans.children()):
            res = dense(res)
            skip_tensors.append(res)
            res = trans(res)

        res = self.middle(res)



        for skip, trans, dense in zip(reversed(skip_tensors), self.up_trans.children(), self.up_dense.children()):
            res = trans(res, skip)
            res = dense(res)

        res= self.final(res)
        t1= torch.transpose(res, -1, -2)
        t2= (t1 + res)


        t3= self.Norm(t2)
        t3= t3.squeeze(1)


        return t3