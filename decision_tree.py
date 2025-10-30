import numpy as np
import logging 

logging.basicConfig(filename='.logs', filemode='w', format='%(asctime)s - %(message)s', encoding='utf-8', level=logging.INFO)


class Node:
    def __init__(self, feature=None, treshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.treshold = treshold 
        self.left = left 
        self.right = right 
        self.value = value 
    
    def is_leaf(self):
        return self.value is not None
    
    def __str__(self):
        return f'feature: {self.feature} treshold: {self.treshold} value: {self.value}'

class DecisionTree:
    def __init__(self, max_depth=100, min_samples_split=2, criterion='gini'):
        self.max_depth = max_depth
        self.min_saples_split = min_samples_split
        self.criterion = criterion
        self.root = None
    
    def _gini(self, y):
        if len(y) == 0:
            return 0
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities**2)
    
    def _entropy(self, y):
        if len(y) == 0:
            return 0
        counts = np.bincount(y)
        probabilities = counts / len(y)
        probabilities = probabilities[probabilities > 0]
        # logging.info(f'{np.log2(probabilities)} {counts} {probabilities}')
        return -np.sum(probabilities * np.log2(probabilities))
    
    def _information_gain(self, y, y_left, y_right, criterion):
        if criterion == 'gini':
            impurity_func = self._gini
        elif criterion == 'entropy':
            impurity_func = self._entropy 
        
        parent_impurity = impurity_func(y)

        # Weighted average of child impurities
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)

        if n == 0:
            return 0
        
        child_impurity = n_left / n * impurity_func(y_left) + n_right / n * impurity_func(y_right)

        return parent_impurity - child_impurity
    
    def _best_split(self, X, y):
        best_gain = -1
        best_treshold = None
        best_feature = None

        n_samples, n_features = X.shape

        # logging.info(f'_best_split: {n_samples} {self.min_saples_split}')

        if n_samples < self.min_saples_split:
            return best_feature, best_treshold

        for feature_idx in range(n_features):
            feature_values = np.unique(X[:, feature_idx])

            # logging.info(f'_best_split next_feature: feature_values_len {len(feature_values) - 1}')

            for i in range(len(feature_values) - 1):
                treshold = (feature_values[i] + feature_values[i + 1]) / 2

                left_mask = X[:, feature_idx] <= treshold
                right_mask = X[:, feature_idx] > treshold

                y_left = y[left_mask]
                y_right = y[right_mask]

                # logging.info(f'_best_split treshold: {treshold} {feature_values}')

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                gain = self._information_gain(y, y_left, y_right, criterion=self.criterion)

                # logging.info(f'_best_split iter: {gain} {best_gain} {treshold}')

                if best_gain < gain:
                    best_gain = gain
                    best_treshold = treshold
                    best_feature = feature_idx
        # logging.info(f'_best_split: {best_feature} {best_treshold}')
        
        return best_feature, best_treshold
    
    def _build_tree(self, X, y, depth=0) -> Node:
        n_samples, n_features = X.shape

        if depth >= self.max_depth or n_samples < self.min_saples_split or len(np.unique(y)) == 1:
            values, counts = np.unique(y, return_counts=True)
            index_of_mode = np.argmax(counts)
            most_common = values[index_of_mode]
            return Node(value=most_common)
        
        best_feature, treshold = self._best_split(X, y)

        # logging.info(str(best_feature) + ' ' + str(treshold))

        if best_feature is None:
            values, counts = np.unique(y, return_counts=True)
            index_of_mode = np.argmax(counts)
            most_common = values[index_of_mode]
            return Node(value=most_common)

        left_mask = X[:, best_feature] <= treshold
        right_mask = X[:, best_feature] > treshold

        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth=depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth=depth + 1)

        return Node(treshold=treshold, feature=best_feature, left=left_subtree, right=right_subtree)

    def fit(self, X, y):
        self.root = self._build_tree(X, y)
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]

    def _predict_single(self, x, node):
        if node.is_leaf():
            return node.value
        
        # logging.info(f'{node}')
        # logging.info(f'{node.feature} {x}')
        
        if x[node.feature] <= node.treshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)
        
    
    def predict(self, X):
        return np.array([self._predict_single(x, self.root) for x in X])
