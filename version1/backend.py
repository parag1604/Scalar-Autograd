"""
version1: a Variable that remembers how it was computed (the compute graph).

What this version teaches
- Wrap every number in a `Variable`. Python's operator overloading
  (__add__, __mul__, ...) lets us write x * y + z as usual, but every result
  also stores its inputs (_prev) and the operation (_op) that produced it.
- Following _prev backwards from the output gives a directed acyclic graph:
  the compute graph. Backpropagation (next versions) walks this graph.
- Only +, * and ** are "primitive"; -, / and the reflected versions
  (4 * x, 5 + x, ...) are built from them, so later we only need derivatives
  for a handful of operations.

New compared to the previous version: this is the first version (forward only,
no gradients yet).
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

        self._prev = set(_children)  # the Variables this one was computed from
        self._op = _operator         # the operation that produced it
        self.label = label           # a name, only used for drawing

    def __add__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.data + other.data, (self, other), '+')
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.data * other.data, (self, other), '*')
        return out

    def __pow__(self, other):
        # only constant (int/float) exponents, which is all we need for x^2
        # and 1/x; the exponent is therefore not part of the graph
        assert isinstance(other, (int, float)), "only int/float powers"
        out = Variable(self.data ** other, (self,), f'^{other}')
        return out

    def __neg__(self):  # -self
        return self * -1

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __truediv__(self, other):  # self / other
        return self * other ** -1

    # reflected operators: called for 4 * x, 5 + x, ... where the left
    # operand is a plain number that does not know about Variables
    def __radd__(self, other):  # other + self
        return self + other

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __rtruediv__(self, other):  # other / self
        return other * (self ** -1)

    def __repr__(self) -> str:
        return f"Variable({self.data})"


def main() -> None:
    x = Variable(2, label='x')
    y = Variable(3, label='y')
    z = Variable(1, label='z')

    print('x =', x)
    print('y =', y)
    print('z =', z)

    # this is x.__mul__(y).__add__(z)
    i = x * y
    i.label = 'i'
    o = i + z
    o.label = 'o'
    print('x * y + z =', o)

    # mixing Variables and plain numbers on either side
    print('x * 4 + z =', x * 4 + z)
    print('4 * x + z =', 4 * x + z)   # needs __rmul__
    print('x * y + 4 =', x * y + 4)
    print('4 + x * y =', 4 + x * y)   # needs __radd__
    print('5 - x / y =', 5 - x / y)   # needs __rsub__, __truediv__, __pow__
    print('4 * 5 + 6 =', 4 * 5 + 6, '(plain Python numbers, no Variable)')

    print('The children of o are:', o._prev, 'via', repr(o._op))
    render_graph(o, 'graph_version1', requires_grad=False)


if __name__ == '__main__':
    main()
