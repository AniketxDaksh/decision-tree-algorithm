import numpy as np
from sklearn.datasets import load_iris

# Load the Iris dataset
data = load_iris()
X, y = data.data, data.target

# Split dataset into training and testing sets (80-20 split)
def train_test_split(X, y, test_size=0.2):
    indices = np.random.permutation(len(X))
    test_count = int(len(X) * test_size)
    return X[indices[test_count:]], X[indices[:test_count]], y[indices[test_count:]], y[indices[:test_count]]

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Define the Decision Tree class
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._predict(x, self.tree) for x in X])

    def _gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return 1 - np.sum(p**2)

    def _split(self, X, y, feature, threshold):
        left = y[X[:, feature] <= threshold]
        right = y[X[:, feature] > threshold]
        return left, right

    def _find_best_split(self, X, y):
        best_gini, best_split = float('inf'), None
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left, right = self._split(X, y, feature, threshold)
                gini = (len(left) * self._gini(left) + len(right) * self._gini(right)) / len(y)
                if gini < best_gini:
                    best_gini, best_split = gini, (feature, threshold)
        return best_split

    def _build_tree(self, X, y, depth=0):
        if len(np.unique(y)) == 1 or (self.max_depth and depth >= self.max_depth):
            return {'leaf': np.bincount(y).argmax()}
        feature, threshold = self._find_best_split(X, y)
        if feature is None:
            return {'leaf': np.bincount(y).argmax()}
        left_idx = X[:, feature] <= threshold
        return {
            'feature': feature, 'threshold': threshold,
            'left': self._build_tree(X[left_idx], y[left_idx], depth + 1),
            'right': self._build_tree(X[~left_idx], y[~left_idx], depth + 1)
        }

    def _predict(self, x, tree):
        if 'leaf' in tree:
            return tree['leaf']
        feature, threshold = tree['feature'], tree['threshold']
        if x[feature] <= threshold:
            return self._predict(x, tree['left'])
        return self._predict(x, tree['right'])

# Train and evaluate the Decision Tree
tree = DecisionTree(max_depth=3)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.2f}")
