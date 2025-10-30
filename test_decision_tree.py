import unittest 
import numpy as np 
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from decision_tree import DecisionTree 


class TestDecisionTreeIris(unittest.TestCase):
    
    def setUp(self):
        iris = load_iris()
        self.X = iris.data
        self.y = iris.target

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
    
    # testing implementation details
    def test_gini_pure(self):
        y = np.array([1, 1, 1])
        tree = DecisionTree()
        self.assertAlmostEqual(tree._gini(y), 0.0)
    
    def test_gini_mixed(self):
        y = np.array([0, 1, 0, 1])
        tree = DecisionTree()
        self.assertAlmostEqual(tree._gini(y), 0.5)
    
    def test_entropy_pure(self):
        y = np.array([1, 1, 1])
        tree = DecisionTree()
        self.assertAlmostEqual(tree._entropy(y), 0.0)
    
    def test_entropy_max(self):
        y = np.array([0, 1, 0, 1])
        tree = DecisionTree()
        self.assertAlmostEqual(tree._entropy(y), 1.0)
    
    def test_information_gain_gini(self):
        tree = DecisionTree()
        y = np.array([0, 0, 1, 1])
        y_left = np.array([0, 0])
        y_right = np.array([1, 1])
        information_gain = tree._information_gain(y, y_left, y_right, criterion='gini')
        self.assertEqual(information_gain, 0.5)
    
    def test_information_gain_entropy(self):
        tree = DecisionTree()
        y = np.array([0, 0, 1, 1])
        y_left = np.array([0, 1])
        y_right = np.array([0, 1])
        information_gain = tree._information_gain(y, y_left, y_right, criterion='entropy')
        self.assertEqual(information_gain, 0.0)
    
    # end of implemantation details tests
    
    def test_iris_gini_accuracy(self):
        tree = DecisionTree(max_depth=4, min_samples_split=2, criterion='gini')
        tree.fit(self.X_train, self.y_train)
        y_pred = tree.predict(self.X_test)
        score = accuracy_score(self.y_test, y_pred, normalize=True)
        self.assertGreaterEqual(score, 0.9)

    
    def test_iris_entropy_accuracy(self):
        tree = DecisionTree(max_depth=4, min_samples_split=2, criterion='entropy')
        tree.fit(self.X_train, self.y_train)
        y_pred = tree.predict(self.X_test)
        score = accuracy_score(self.y_test, y_pred, normalize=True)
        self.assertGreaterEqual(score, 0.9)
    
    def test_full_fit_perfect_train_score(self):
        tree = DecisionTree(max_depth=1000000, min_samples_split=2)
        tree.fit(self.X_train, self.y_train)
        y_pred = tree.predict(self.X_test)
        score = accuracy_score(self.y_test, y_pred)
        self.assertEqual(score, 1.0)
    
    def test_best_split_is_used_correctly(self):
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([0, 0, 1, 1, 1])
        tree = DecisionTree(max_depth=1)
        tree.fit(X, y)

        root = tree.root
        self.assertEqual(root.feature, 0)
        self.assertGreater(root.treshold, 2)
        self.assertLess(root.treshold, 3)

    def test_predict_output_shape(self):
        tree = DecisionTree()
        tree.fit(self.X_train, self.y_train)
        y_pred = tree.predict(self.X_test)
        self.assertEqual(self.y_test.shape, y_pred.shape)
    
    def test_leaf_values_are_valid_classes(self):
        tree = DecisionTree(max_depth=3)
        tree.fit(self.X_train, self.y_train)
        y_pred = tree.predict(self.X_test)
        unique_preds = np.unique(y_pred)
        valid_classes = np.unique(self.y_train)
        for pred in unique_preds:
            self.assertIn(pred, valid_classes)
    
    def test_reproducibility(self):
        tree1 = DecisionTree()
        tree2 = DecisionTree()
        tree1.fit(self.X_train, self.y_train)
        tree2.fit(self.X_train, self.y_train)
        pred1 = tree1.predict(self.y_test)
        pred2 = tree2.predict(self.y_test)
        np.testing.assert_array_equal(pred1, pred2)

if __name__ == '__main__':
    unittest.main()
