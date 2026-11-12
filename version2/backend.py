"""
version2: gradients by hand with the chain rule, checked numerically.

What this version teaches
- Every Variable gets a `.grad` slot holding d(output)/d(this variable).
- For o = d * f with f = e + c and e = a * b, the chain rule gives
      do/do = 1
      do/df = d,           do/dd = f
      do/de = do/df * 1,   do/dc = do/df * 1        (+ passes gradients on)
      do/da = do/de * b,   do/db = do/de * a        (* swaps the inputs)
  i.e. each node's gradient = (local derivative) x (gradient flowing in from
  the output side). We fill these in by hand, from the output backwards.
- To make sure we got them right we compare with the numerical derivative
  (f(v + h) - f(v - h)) / 2h of the same expression for every input v.

New compared to version1: the .grad attribute, manual backpropagation and the
numerical gradient check. Next version automates the manual part.
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
        self.grad = 0.  # d(output)/d(self), filled in by backpropagation

        self._prev = set(_children)
        self._op = _operator
        self.label = label

    def __add__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.data + other.data, (self, other), '+')
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.data * other.data, (self, other), '*')
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only int/float powers"
        out = Variable(self.data ** other, (self,), f'^{other}')
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


def expression(a, b, c, d):
    """o = d * (a * b + c); works for Variables and plain floats alike."""
    return d * (a * b + c)


def num_grad(values: dict, name: str, h: float = 1e-4) -> float:
    """Central-difference estimate of do/d(name) at the given input values."""
    plus, minus = dict(values), dict(values)
    plus[name] += h
    minus[name] -= h
    return (expression(**plus) - expression(**minus)) / (2 * h)


def main() -> None:
    a = Variable(3, label='a')
    b = Variable(-2, label='b')
    c = Variable(1, label='c')
    d = Variable(2, label='d')

    # forward pass
    e = a * b
    e.label = 'e'
    f = e + c
    f.label = 'f'
    o = d * f
    o.label = 'o'

    # gradient backpropagation, by hand, from the output backwards
    o.grad = 1.

    f.grad = d.data * o.grad
    d.grad = f.data * o.grad

    c.grad = f.grad
    e.grad = f.grad

    a.grad = b.data * e.grad
    b.grad = a.data * e.grad

    # check every input against the numerical derivative
    values = dict(a=a.data, b=b.data, c=c.data, d=d.data)
    for var in (a, b, c, d):
        print(f'do/d{var.label}: backprop = {var.grad:+.4f} | '
              f'numerical = {num_grad(values, var.label):+.4f}')

    render_graph(o, 'graph_version2')


if __name__ == '__main__':
    main()
