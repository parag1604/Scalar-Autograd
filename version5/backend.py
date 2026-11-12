"""
version5 engine: unchanged from version4 (Variable with +, *, **, sigmoid and
backward). What is new in version5 lives in nn.py and main.py.
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
