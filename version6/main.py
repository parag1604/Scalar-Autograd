"""
version6: binary classification on real data (Iris) with the cross-entropy loss.

What this version teaches
- For classification the network outputs p = P(y = 1 | x) through a sigmoid,
  and we minimise the binary cross-entropy (negative log-likelihood)
      loss = -log(p)       if y = 1
      loss = -log(1 - p)   if y = 0.
  To build it from Variables the engine needs a `log` node (backend.py).
- Real data needs a little care: shuffle every epoch (the Iris file is sorted
  by class, and SGD on sorted data first learns "always 0", then "always 1"),
  standardise the features, and keep a held-out test split to measure
  generalisation rather than memorisation.
- We use the first 100 Iris flowers: setosa (0) vs versicolor (1), 4 features.
  sklearn's load_iris ships with the package, so nothing is downloaded.

New compared to version5: log/exp/tanh/ReLU nodes, the BCE loss, a real
dataset with shuffling, standardisation and a train/test split.
"""
import argparse
import random

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris

from version6.nn import MLP

EPS = 1e-7  # keeps log() away from log(0) if the sigmoid saturates


def bce_loss(y_pred, y):
    if y == 1:
        return -((y_pred + EPS).log())
    return -((1 + EPS - y_pred).log())


def evaluate(model, X, Y):
    total_loss, total_acc = 0., 0.
    for x, y in zip(X, Y):
        y_pred = model(x)
        total_loss += bce_loss(y_pred, y).data
        total_acc += (y_pred.data > 0.5) == y
    return total_loss / len(X), total_acc / len(X)


def plot_decision_boundary(clf, X, Y, cmap='Paired_r'):
    """
    The model sees 4 features but a plot shows 2: we sweep the two petal
    features (columns 2, 3) and hold the sepal features at their mean (0
    after standardisation).
    """
    h = 0.05
    x_min, x_max = X[:, 2].min() - 0.5, X[:, 2].max() + 0.5
    y_min, y_max = X[:, 3].min() - 0.5, X[:, 3].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = []
    for x, y in zip(xx.ravel(), yy.ravel()):
        Z.append(clf([0., 0., x, y]).data > 0.5)
    Z = np.array(Z).reshape(xx.shape)

    plt.figure(figsize=(5, 5))
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.25)
    plt.contour(xx, yy, Z, colors='k', linewidths=0.7)
    plt.scatter(X[:, 2], X[:, 3], c=Y, cmap=cmap, edgecolors='k')
    plt.xlabel('petal length (standardised)')
    plt.ylabel('petal width (standardised)')
    plt.title('Setosa (0) vs Versicolor (1)')
    plt.show()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description='Scalar-autograd MLP on Iris')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--activation', default='sigmoid',
                        choices=['sigmoid', 'tanh', 'relu'],
                        help='hidden-layer activation')
    parser.add_argument('--seed', type=int, default=2023)
    args = parser.parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)

    model = MLP(4, [4, 4, 1], hidden_activation=args.activation)
    print(model)

    # load iris dataset: first 100 rows = classes 0 and 1
    iris = load_iris()
    X = iris.data[:100]
    Y = iris.target[:100]

    # 80/20 train/test split, then standardise with TRAIN statistics only
    perm = np.random.permutation(len(X))
    train_idx, test_idx = perm[:80], perm[80:]
    mean, std = X[train_idx].mean(axis=0), X[train_idx].std(axis=0)
    X = (X - mean) / std
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    # training loop
    for epoch in range(args.epochs):

        total_acc = 0
        total_loss = 0
        for i in np.random.permutation(len(X_train)):  # shuffle every epoch
            x, y = X_train[i], Y_train[i]

            # forward pass
            y_pred = model(x)

            # calculate loss (BCE)
            loss = bce_loss(y_pred, y)

            # zero gradients
            model.zero_grad()

            # backward pass
            loss.backward()

            # update weights using SGD
            for p in model.parameters():
                p.data -= args.lr * p.grad

            total_loss += loss.data
            total_acc += (y_pred.data > 0.5) == y
        total_acc /= len(X_train)
        total_loss /= len(X_train)

        print(f'Epoch: {epoch + 1} | Train Loss: {total_loss:.4f} | '
              f'Train Acc: {total_acc:.4f}')

    test_loss, test_acc = evaluate(model, X_test, Y_test)
    print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}')

    # plot decision boundary
    plot_decision_boundary(model, X, Y)


if __name__ == '__main__':
    main()
