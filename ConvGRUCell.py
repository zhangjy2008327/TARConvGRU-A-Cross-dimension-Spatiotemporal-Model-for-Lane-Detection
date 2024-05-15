#!/usr/bin/evn python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 - zihao.chen <zihao.chen@moji.com>
'''
Author: zihao.chen
Create Date: 2018-03-29
Modify Date: 2018-03-29
descirption: ""
'''
# best testing------------- __test_acc=98.08212159607241__precision=0.8841138859937758__recall=0.9437325979911038__F1_measure=0.9129509527845213_epoch=35_98.38638305664062.pth
import torch
from torch import nn
# import torch.nn.functional as f
# import numpy as np
from torch.autograd import Variable
from triplet_attention import TripletAttention

class ConvGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.height, self.width = 32, 64
        self.dropout = nn.Dropout(p=0.5)
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.ConvGates = nn.Conv2d(self.input_size + self.hidden_size,
                                   2 * self.hidden_size,
                                   self.kernel_size,
                                   padding=self.padding
                                   )
        self.Conv_ct = nn.Conv2d(self.input_size + self.hidden_size, self.hidden_size, self.kernel_size,
                                 padding=self.padding)
        self.ta = TripletAttention()
        self.rta = TripletAttention()
    def forward(self, input, hidden):
        if hidden is None:
            
            size_h = [input.data.size()[0], self.hidden_size] + list(input.data.size()[2:])
            hidden = Variable(torch.zeros(size_h).cuda())
        if input is None:
            size_h = [hidden.data.size()[0], self.input_size] + list(hidden.data.size()[2:])
            input = Variable(torch.zeros(size_h).cuda())
        input = self.rta(input)
        hidden = self.rta(hidden)
        c1 = self.ConvGates(torch.cat((input, hidden), 1))
        (rt, ut) = c1.chunk(2, 1)
        reset_gate = self.dropout(torch.sigmoid(rt))

        update_gate = self.dropout(torch.sigmoid(ut))
        update_gate = self.ta(update_gate)
        update_gate = torch.sigmoid(update_gate + ut + rt)
        gated_hidden = torch.mul(reset_gate, hidden)
        p1 = self.Conv_ct(torch.cat((input, gated_hidden), 1))
        ct = torch.tanh(p1)
        next_h = torch.mul(update_gate, ct) + (1 - update_gate) * hidden
        return next_h, 0

    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_size, self.height, self.width).cuda(),
                torch.zeros(batch_size, self.hidden_size, self.height, self.width).cuda())

class ConvGRU(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvGRU, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvGRUCell(input_size=self.input_dim,
                                         hidden_size= self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i]))
                                         )

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output

        -------
        here data = [5, 1, 512, 8, 16]= input_tensor is passed into ConvLSTM
        5 is number of times, each time is one batch, as 1,
        and each batch_size size has 512 feature maps, each feature map size is 8x16
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            output_inner1 = []
            final_out = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input=cur_layer_input[:, t, :, :, :],
                                                 hidden=h)
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            layer_output = layer_output.permute(1, 0, 2, 3, 4)
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    """
    check the type of kernel_size to be the same

    """

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
