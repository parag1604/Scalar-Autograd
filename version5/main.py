"""
version5: a multi-layer perceptron built from scalar Variables learns XOR.

What this version teaches
- With an engine that can differentiate any expression, a neural network is
  just a bigger expression. nn.py packages Neurons into Layers and Layers into
  an MLP, and collects all weights with .parameters().
- The training loop is the same five steps as in every framework:
  forward pass -> loss -> zero_grad -> backward -> gradient-descent update
  (p.data -= lr * p.grad for every parameter).
- Here we do one update per data point (stochastic gradient descent) on the
  four XOR points with the squared-error loss.

New compared to version4: the nn.py library (Module, Neuron, Layer, MLP) and a
training loop.
"""
import argparse
import random

import numpy as np
from matplotlib import pyplot as plt

from version5.nn import MLP


def plot_decision_boundary(clf, X, Y, title, cmap='Paired_r'):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 10 * h, X[:, 0].max() + 10 * h
    y_min, y_max = X[:, 1].min() - 10 * h, X[:, 1].max() + 10 * h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = []
    for x, y in zip(xx.ravel(), yy.ravel()):
        Z.append(clf([x, y]).data > 0.5)
    Z = np.array(Z).reshape(xx.shape)

    plt.figure(figsize=(5, 5))
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.25)
    plt.contour(xx, yy, Z, colors='k', linewidths=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap, edgecolors='k')
    plt.title(title)
    plt.show()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description='Scalar-autograd MLP on XOR')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=2023)
    args = parser.parse_args(argv)
    random.seed(args.seed)

    model = MLP(2, [3, 1])
    print(model)

    X = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]

    Y = [
        0,
        1,
        1,
        0,
    ]

    # training loop
    for epoch in range(args.epochs):

        total_loss = 0

        for x, y in zip(X, Y):

            # forward pass
            y_pred = model(x)

            # calculate loss
            loss = (y - y_pred) ** 2

            # zero gradients
            model.zero_grad()

            # backward pass
            loss.backward()

            # update weights using SGD
            for p in model.parameters():
                p.data -= args.lr * p.grad

            total_loss += loss.data

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}: Loss = {total_loss / len(X):.6f}")

    # print predictions
    for x, y in zip(X, Y):
        y_pred = model(x)
        print(f"{x} -> {y_pred.data:.4f} (target {y})")

    # plot decision boundary
    plot_decision_boundary(model, np.array(X), np.array(Y),
                           title='Scalar-autograd MLP on XOR')


if __name__ == '__main__':
    main()
