import numpy as np


class Evaluator(object):
    def __init__(self, img_height, img_width, fetch_loss_and_grads):
        self.loss_value = None
        self.grad_values = None
        self.img_height = img_height
        self.img_width = img_width
        self.fetch_loss_and_grads = fetch_loss_and_grads

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, self.img_height, self.img_width, 3))
        outs = self.fetch_loss_and_grads([x])
        loss_value = outs[0]
        outs1 = np.array(outs[1])
        grad_values = outs1.flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values
