# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
from collections import OrderedDict

class BatchNorm1d(nn.Module):
	# TODO START
	def __init__(self, num_features, momentum=0.1, eps=1e-5):
		super(BatchNorm1d, self).__init__()
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
		# input: [batch_size, num_feature_map * height * width]
		if self.training:
			mean, var = input.mean(dim=0), input.var(dim=0)
			self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
			self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
		else:
			mean, var = self.running_mean, self.running_var
		
		normalized = (input - mean) / torch.sqrt(var + self.eps)
		return normalized * self.weight + self.bias
	# TODO END

class Dropout(nn.Module):
	# TODO START
	def __init__(self, p=0.5):
		super(Dropout, self).__init__()
		self.p = p

	def forward(self, input):
		if self.training:
			q = 1 - self.p
			return input * torch.bernoulli(torch.full_like(input, q, device=input.device)) / q
		else:
			return input
	# TODO END

class Model(nn.Module):
	def __init__(self, drop_rate=0.5, hidden_size=1024, input_shape=3072, num_classes=10):
		super(Model, self).__init__()
		# TODO START
		self.to_logits = nn.Sequential(OrderedDict([
			('Linear1', nn.Linear(input_shape, hidden_size, bias=True)),
			("BN", BatchNorm1d(hidden_size)),
			("ReLU", nn.ReLU()),
			("Dropout", Dropout(drop_rate)),
			("Linear2", nn.Linear(hidden_size, num_classes, bias=True))
		]))
		# TODO END
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):
		# TODO START
		logits = self.to_logits(x)
		# TODO END

		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return pred
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

		return loss, acc
