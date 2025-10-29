import unittest 
import numpy as np 
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
from decision_tree import DecisionTree 


class TestDecisionTreeIris(unittest.TestCase):
    
    def setUp(self):
        iris = load_iris()
        self.X = iris.data
        self.y = iris.target

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
    
    def test_best_split_is_used_correctly(self):
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([0, 0, 1, 1, 1])
        tree = DecisionTree(max_depth=1)
        tree.fit(X, y)

        root = tree.root
        self.assertEqual(root.feature, 0)
        self.assertGreater(root.treshold, 2)
        self.assertLess(root.treshold, 3)
    
    def test_leaf_values_are_valid_classes(self):
        ...
