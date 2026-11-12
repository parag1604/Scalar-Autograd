"""
version6 nn library: same as version5, but every Neuron takes an `activation`
name ('sigmoid', 'tanh', 'relu' or 'linear'), and the MLP uses one activation
for the hidden layers and another for the output layer (sigmoid here, so the
output is a probability for the cross-entropy loss).
"""
import math
import random

from version6.backend import Variable


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, num_inputs, activation='sigmoid'):
        assert activation in ('sigmoid', 'tanh', 'relu', 'linear')
        # scale the uniform init by 1/sqrt(fan_in) so the weighted sum does
        # not grow with the number of inputs. The scaling is applied to the
        # NUMBER, so each weight is a leaf Variable (not the output of a *).
        self.w = [Variable(random.uniform(-1, 1) / math.sqrt(num_inputs))
                  for _ in range(num_inputs)]
        self.b = Variable(0.)
        self.activation = activation

    def __call__(self, x):
        # sum(..., start=self.b) gives w . x + b
        a = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return a if self.activation == 'linear' else getattr(a, self.activation)()

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{self.activation.capitalize()}Neuron({len(self.w)})"


class Layer(Module):
    def __init__(self, num_inputs, num_outputs, **kwargs):
        self.neurons = [
            Neuron(num_inputs, **kwargs) for _ in range(num_outputs)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    def __init__(self, num_inputs, num_outputs, hidden_activation='sigmoid',
                 output_activation='sigmoid'):
        # num_outputs lists the layer widths, e.g. [4, 4, 1]
        sizes = [num_inputs] + num_outputs
        self.layers = [
            Layer(sizes[i], sizes[i + 1],
                  activation=output_activation if i == len(num_outputs) - 1
                  else hidden_activation)
            for i in range(len(num_outputs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
