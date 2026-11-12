"""
version6 engine: version5's engine plus exp, log, tanh and ReLU.

log is what the cross-entropy loss in main.py needs; tanh and ReLU are the two
most common hidden-layer activations, selectable with --activation.
"""
import math
from typing import Union


class Variable:

    def __init__(
            self,
            value: Union[int, float],
            _children: tuple = (),
            _operator: str = '',
            label: str = '',
    ) -> None:
        self.data = value
        self.grad = 0.

        self._backward = lambda: None  # leaves have nothing to propagate
        self._prev = set(_children)
        self._op = _operator
        self.label = label

    def __add__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only int/float powers"
        out = Variable(self.data ** other, (self,), f'^{other}')
        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1 / (1 + math.exp(-self.data))
        out = Variable(s, (self,), 'σ')
        def _backward():
            # d sigmoid(x)/dx = sigmoid(x) (1 - sigmoid(x))
            self.grad += (1 - out.data) * out.data * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Variable(math.exp(self.data), (self,), 'exp')
        def _backward():
            self.grad += out.data * out.grad  # d e^x/dx = e^x
        out._backward = _backward
        return out

    def log(self):
        out = Variable(math.log(self.data), (self,), 'ln')
        def _backward():
            self.grad += (1 / self.data) * out.grad  # d ln x/dx = 1/x
        out._backward = _backward
        return out

    def tanh(self):
        out = Variable(math.tanh(self.data), (self,), 'tanh')
        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Variable(max(0., self.data), (self,), 'ReLU')
        def _backward():
            # the derivative at exactly 0 is undefined; we use 0 (a convention)
            self.grad += (1. if self.data > 0 else 0.) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other ** -1

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return other * (self ** -1)

    def __repr__(self) -> str:
        return f"Variable({self.data})"

    def backward(self) -> None:
        # topological order: a node is appended only after all its inputs
        topologically_sorted_vertices = []
        visited_nodes = set()
        def build_topological_ordering(node):
            if node not in visited_nodes:
                visited_nodes.add(node)
                for child in node._prev:
                    build_topological_ordering(child)
                topologically_sorted_vertices.append(node)
        build_topological_ordering(self)

        # d(self)/d(self) = 1, then apply the chain rule from output to inputs
        self.grad = 1.
        for node in reversed(topologically_sorted_vertices):
            node._backward()
