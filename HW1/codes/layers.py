import numpy as np

class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable
        self._saved_tensor = None

    def __str__(self):
        return f"{type(self).__name__}(name={self.name})"

    def forward(self, input):
        pass

    def backward(self, grad_output):
        pass

    def update(self, config):
        pass

    def _saved_for_backward(self, tensor):
        '''The intermediate results computed during forward stage
        can be saved and reused for backward, for saving computation'''

        self._saved_tensor = tensor

class Relu(Layer):
    def __init__(self, name, activation_report=None):
        super(Relu, self).__init__(name)
        self.activation_report = activation_report

    def forward(self, input):
        # TODO START
        self.x = input
        self.x[self.x < 0] = 0

        if self.activation_report: # Activation Analysis Hook
            self.activation_report(self)
        return self.x
        # TODO END

    def backward(self, grad_output):
        # TODO START
        return np.where(self.x <= 0, 0, grad_output)
        # TODO END

class Sigmoid(Layer):
    def __init__(self, name, activation_report=None):
        super(Sigmoid, self).__init__(name)
        assert activation_report is None, "Activation for Sigmoid is not well-defined"

    def forward(self, input):
        # TODO START
        self.y = 1 / (np.exp(-input) + 1)
        return self.y.copy()
        # TODO END

    def backward(self, grad_output):
        # TODO START
        return self.y * (1 - self.y) * grad_output
        # TODO END

class Gelu(Layer):
    def __init__(self, name, activation_report=None):
        super(Gelu, self).__init__(name)
        self.activation_report = activation_report

    def forward(self, input):
        # TODO START
        s = np.sqrt(2 / np.pi)
        b = 0.044715 * s
        self.x = input
        self.th = np.tanh(s * self.x + b * self.x ** 3)

        if self.activation_report:  # Activation Analysis Hook
            self.activation_report(self)
        return 0.5 * self.x * (1 + self.th)
        # TODO END
    
    def backward(self, grad_output):
        # TODO START
        s = np.sqrt(2 / np.pi)
        b = 0.044715 * s
        g_grad =  0.5 + 0.5 * self.th + 0.5 * self.x * (1 - self.th ** 2) * (s + 3 * b * self.x ** 2)
        return g_grad * grad_output
        # TODO END

class Linear(Layer):
    def __init__(self, name, in_num, out_num, init_std, norm_report=None):
        super(Linear, self).__init__(name, trainable=True)
        self.in_num = in_num
        self.out_num = out_num
        self.init_std = init_std
        self.W = np.random.randn(in_num, out_num) * init_std
        self.b = np.zeros(out_num)

        self.grad_W = np.zeros((in_num, out_num))
        self.grad_b = np.zeros(out_num)

        self.diff_W = np.zeros((in_num, out_num))
        self.diff_b = np.zeros(out_num)

        self.norm_report = norm_report
    
    def __str__(self):
        return f"{type(self).__name__}(name={self.name}, in_num={self.in_num}, out_num={self.out_num}, init_std={self.init_std})"

    def forward(self, input):
        # TODO START
        '''
        input: (B, in_C)
        return: (B, out_C)
        '''
        self.x = input
        return self.x @ self.W + self.b
        # TODO END

    def backward(self, grad_output):
        # TODO START
        '''
        grad_output: (B, out_C)
        return: (B, in_C)
        '''
        self.grad_W, self.grad_b  = self.x.T @ grad_output, grad_output.sum(axis=0)

        if self.norm_report:  # Gradient Analysis Hook
            self.norm_report(self)

        return grad_output @ self.W.T
        # TODO END

    def update(self, config):
        mm = config['momentum']
        lr = config['learning_rate']
        wd = config['weight_decay']

        self.diff_W = mm * self.diff_W + (self.grad_W + wd * self.W)
        self.W = self.W - lr * self.diff_W

        self.diff_b = mm * self.diff_b + (self.grad_b + wd * self.b)
        self.b = self.b - lr * self.diff_b