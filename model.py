import numpy as np

class LogisticRegression:
    def __init__(self, train_feats, test_feats, train_labels, test_labels, lr=.1, num_its=10000):
        self.train_feats = train_feats
        self.test_feats = test_feats
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.theta = np.zeros((train_feats.shape[1]))
        self.lr = lr
        self.num_its = num_its


    """
    Computes sigmoid function.

    :returns: 1 / (1 + e^-z)
    """
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z.astype(np.float)).astype(np.float))


    """
    Computes optimal weights for the model using batch gradient descent.
    """
    def train(self):
        for i in range(self.num_its):
            a = self.sigmoid(np.dot(self.train_feats, self.theta))
            z = a - self.train_labels
            dw = np.dot(z, self.train_feats)
            theta = self.theta.copy()
            theta = theta - (self.lr * dw)
            self.theta = theta.copy()


    """
    Tests model against the test set, and prints the model's accuracy.
    """
    def test(self):
        preds = self.sigmoid(np.dot(self.test_feats, self.theta))
        acc = ((preds == self.test_labels).sum()) / len(self.test_labels)
        acc *= 100
        print(f'Accuracy: {acc}%')