from __future__ import division
import numpy as np

class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # TODO START
        '''
        input, target: (B, C)
        '''
        self.delta = input - target
        return 0.5 * np.mean(np.sum(self.delta ** 2, axis=1))
        # TODO END

    def backward(self, input, target):
		# TODO START
        return self.delta / self.delta.shape[0]
		# TODO END


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name, threshold=1e-10):
        self.name = name
        self.threshold = threshold

    def forward(self, input, target):
        # TODO START
        '''
        input, target: (B, C)
        '''
        i_max = input.max(axis=1)[:, None]
        exp = np.exp(input - i_max)
        exp_sum = exp.sum(axis=1)[:,None]
        exp /= exp_sum
        exp += self.threshold
        self.probabilities = exp
        log_v = np.log(self.probabilities)
        ce = -(log_v * target).sum(axis=1)
        return ce.mean()
        # TODO END

    def backward(self, input, target):
        # TODO START
        return (self.probabilities - target) / target.shape[0]
        # TODO END

class HingeLoss(object):
    def __init__(self, name, margin):
        self.name = name
        self.margin = margin

    def forward(self, input, target):
        # TODO START 
        '''
        input, target: (B, C)
        '''
        labels = np.argmax(target, axis=1)
        target_values = np.take_along_axis(input, labels[:, None], axis=1)
        self.hinge = self.margin - target_values + input
        self.hinge[(target == 1) | (self.hinge < 0)] = 0
        return self.hinge.sum(axis=1).mean()
        # TODO END

    def backward(self, input, target):
        # TODO START
        grad = np.where(self.hinge > 0, 1, 0)
        grad[target == 1] = -grad.sum(axis=1)
        grad = grad / target.shape[0]
        return grad
        # TODO END
