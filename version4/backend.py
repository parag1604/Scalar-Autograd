"""
version4: a single artificial neuron, sigmoid(w1 x1 + w2 x2 + b).

What this version teaches
- A neuron is just a small expression graph: a weighted sum of its inputs plus
  a bias, followed by a non-linear activation. Here the activation is the
  sigmoid, sigma(z) = 1 / (1 + e^-z).
- Any differentiable function can be added to the engine as ONE new node, as
  long as we provide its local derivative. For the sigmoid it is
  sigma(z) (1 - sigma(z)), which we can compute from the output alone.
- After a.backward(), w1.grad, w2.grad and b.grad tell us how to change the
  weights to change the neuron's output. That is all training needs.

New compared to version3: the sigmoid operation and the neuron example.
"""
import math
from typing import Union

from common.utils import render_graph


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



def main() -> None:
    w1 = Variable(3, label='w1')
    w2 = Variable(-2, label='w2')
    b = Variable(-1.0375, label='b')

    x1 = Variable(4, label='x1')
    x2 = Variable(5, label='x2')

    # forward pass: z = w1 x1 + w2 x2 + b,  a = sigmoid(z)
    h1 = w1 * x1
    h1.label = 'h1'
    h2 = w2 * x2
    h2.label = 'h2'

    i = h1 + h2
    i.label = 'i'
    o = i + b
    o.label = 'o'

    a = o.sigmoid()
    a.label = 'a'

    # gradient backpropagation
    a.backward()
    print(f'neuron output a = {a.data:.4f}')
    for var in (w1, w2, b, x1, x2):
        print(f'da/d{var.label} = {var.grad:+.4f}')

    render_graph(a, 'graph_version4')


if __name__ == '__main__':
    main()
