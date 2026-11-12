"""
version3: automatic backpropagation (reverse-mode autodiff).

What this version teaches
- Each operation now also creates a small closure `_backward` that knows its
  own local derivative and pushes the incoming gradient (out.grad) to its
  inputs, exactly the rules we applied by hand in version2:
      + :  self.grad += out.grad,             other.grad += out.grad
      * :  self.grad += other.data * out.grad, other.grad += self.data * out.grad
      **:  self.grad += n * self.data**(n-1) * out.grad
- `backward()` sorts the graph topologically (every node after all of its
  inputs), seeds the output with grad = 1, and calls the closures in reverse
  order, so a node's gradient is complete before it is passed further down.
- Gradients are ACCUMULATED with +=, not assigned. A Variable that is used
  twice (y = x + x) receives a contribution from each use (the multivariate
  chain rule sums over paths). The demo at the end checks this.

New compared to version2: _backward closures, topological sort and
Variable.backward(), so no more manual chain rule.
"""
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
    a = Variable(3, label='a')
    b = Variable(-2, label='b')
    c = Variable(1, label='c')
    d = Variable(2, label='d')

    # forward pass (same expression as version2)
    e = a * b
    e.label = 'e'
    f = e + c
    f.label = 'f'
    o = f * d
    o.label = 'o'

    # gradient backpropagation: one call
    o.backward()
    for var in (a, b, c, d, e, f):
        print(f'do/d{var.label} = {var.grad:+.4f}')

    # why +=: x is used twice, dy/dx must be 2 (it would be 1 with =)
    x = Variable(3, label='x')
    y = x + x
    y.backward()
    print(f'y = x + x  ->  dy/dx = {x.grad} (expected 2.0)')

    render_graph(o, 'graph_version3')


if __name__ == '__main__':
    main()
