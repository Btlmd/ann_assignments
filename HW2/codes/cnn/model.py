# -*- coding: utf-8 -*-

import imp
from typing import OrderedDict
import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
class BatchNorm2d(nn.Module):
	# TODO START
	def __init__(self, num_features, momentum=0.1, eps=1e-5):
		super(BatchNorm2d, self).__init__()
		self.num_features = num_features

		# Parameters
		self.weight = Parameter(torch.empty(self.num_features))
		self.bias = Parameter(torch.empty(self.num_features))

		# Store the average mean and variance
		self.register_buffer('running_mean', torch.zeros(self.num_features))
		self.register_buffer('running_var', torch.ones(self.num_features))
		
		# Initialize your parameter
		init.ones_(self.weight)
		init.zeros_(self.bias)

		self.momentum = momentum
		self.eps = eps

	def forward(self, input):
		# input: [batch_size, num_feature_map, height, width]
		if self.training:
			mean, var = input.mean(dim=(0, 2, 3)), input.var(dim=(0, 2, 3))
			self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
			self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
		else:
			mean, var = self.running_mean, self.running_var
		
		normalized = (input - mean.view(1, self.num_features, 1, 1)) / torch.sqrt(var.view(1, self.num_features, 1, 1) + self.eps)
		return normalized * self.weight.view(1, self.num_features, 1, 1) + self.bias.view(1, self.num_features, 1, 1)
	# TODO END

	def __repr__(self):
		return f"BatchNorm2d(num_features={self.num_features}, momentum={self.momentum}, eps={self.eps})"

class Dropout(nn.Module):
	# TODO START
	def __init__(self, p=0.5, align_channel=True):
		super(Dropout, self).__init__()
		self.p = p
		self.align_channel = align_channel

	def forward(self, input):
		assert len(input.shape) == 4
		if self.training:
			q = 1 - self.p
			if self.align_channel:
				co = (torch.bernoulli(torch.full(input.shape[:2], q, device=input.device)) / q).view(*input.shape[:2], 1, 1)
			else:
				co = torch.bernoulli(torch.full_like(input, q, device=input.device)) / q
			return input * co
		else:
			return input
	# TODO END

	def __repr__(self):
		return f"Dropout(p={self.p}, align_channel={self.align_channel})"

class Model(nn.Module):
	def __init__(
		self, 
		conv_ch, 
		conv_ker, 
		pool_ker, 
		pool_stride, 
		drop_rate, 
		num_classes=10, 
		input_width=32,
		channel_align=True,
		):
		super(Model, self).__init__()
		# TODO START
		
		assert len(conv_ch) == 2
		assert len(conv_ker) == 2
		assert len(pool_ker) == 2
		assert len(pool_stride) == 2
		assert len(drop_rate) == 2

		self.conv0 = nn.Sequential(OrderedDict([
			("Conv", nn.Conv2d(3, conv_ch[0], conv_ker[0], padding=conv_ker[0] // 2)),  
			("BN", BatchNorm2d(conv_ch[0])),
			("ReLU", nn.ReLU()),
			("Dropout", Dropout(drop_rate[0], channel_align)),
			("MaxPool", nn.MaxPool2d(pool_ker[0], pool_stride[0], padding=pool_ker[0] // 2))  
		]))
		C = input_width + (conv_ker[0] // 2) * 2 - conv_ker[0] + 1
		C = (C - pool_ker[0] + (pool_ker[0] // 2) * 2) // pool_stride[0] + 1

		self.conv1 = nn.Sequential(OrderedDict([
			("Conv", nn.Conv2d(conv_ch[0], conv_ch[1], conv_ker[1], 
			padding=conv_ker[1] // 2
			)),  
			("BN", BatchNorm2d(conv_ch[1])),
			("ReLU", nn.ReLU()),
			("Dropout", Dropout(drop_rate[1], channel_align)),
			("MaxPool", nn.MaxPool2d(pool_ker[1], pool_stride[1], padding=pool_ker[1] // 2))  
		]))
		C = C + (conv_ker[1] // 2) * 2 - conv_ker[1] + 1
		C = (C - pool_ker[1] + (pool_ker[1] // 2) * 2) // pool_stride[1] + 1

		self.fc = nn.Linear(conv_ch[1] * C * C, num_classes) 
		# TODO END
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):
		# TODO START
		x = self.conv0(x)
		x = self.conv1(x)
		x = x.flatten(start_dim=1)
		logits = self.fc(x)
		# TODO END

		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return pred
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

		return loss, acc
