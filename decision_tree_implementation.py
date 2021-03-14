from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib
import copy

def gini(x):
    p = sum(x) / len(x)
    return 2 * p * (1 - p)
    
def entropy(x):
    p = sum(x) / len(x)
    return -p * log2(p) - (1 - p) * log2(1 - p)
    
def gain(left_y, right_y, criterion):
    y = np.concatenate((left_y, right_y), axis=0)
    before_split = criterion(y)
    after_split = (len(left_y) * criterion(left_y) + len(right_y) * criterion(right_y)) / len(y)
    return before_split - after_split

class DecisionTreeLeaf:
    def __init__(self, y):
        self.y = y

class DecisionTreeNode:
    def __init__(self, split_dim, split_value, left, right):
        self.split_dim = split_dim
        self.split_value = split_value
        self.left = left
        self.right = right

class DecisionTreeClassifier:
    def __init__(self, criterion="gini", max_depth=None, min_samples_leaf=1):
        if criterion == 'gini':
            self.criterion = gini
        else:
            self.criterion = entropy
        self.root = None
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
    
    def dfs(self, X, y, depth):
        if len(np.unique(y)) == 1 or len(y) <= self.min_samples_leaf or depth == self.max_depth: 
            return DecisionTreeLeaf(y)

        max_gain = -10e8
        max_split_val = None 
        max_gain_dim = None 
        for i, dim_str in enumerate(X.columns):
            dim = X[dim_str]
            split_val = np.random.choice(dim) 
            left_y = y.loc[dim < split_val]
            right_y = y.loc[dim >= split_val]
            if any(left_y) and any(right_y):
                cur_gain = gain(left_y, right_y, self.criterion)
                if cur_gain > max_gain:
                    max_gain, max_split_val, max_gain_dim, max_gain_index = cur_gain, split_val, dim_str, i    

        if not max_split_val:
            return DecisionTreeLeaf(y)
        
        left_y = y.loc[X[max_gain_dim] < max_split_val]
        left_X = X.loc[X[max_gain_dim] < max_split_val]
        left_tree = self.dfs(left_X, left_y, depth + 1)
        
        right_y = y.loc[X[max_gain_dim] >= max_split_val]
        right_X = X.loc[X[max_gain_dim] >= max_split_val]
        right_tree = self.dfs(right_X, right_y, depth + 1)
        
        return DecisionTreeNode(max_gain_index, max_split_val, left_tree, right_tree)
    
    def fit(self, X, y):
        self.root = self.dfs(X, y, 0)
        
    def predict_instance(self, cur_root, x):
        if isinstance(cur_root, DecisionTreeLeaf):
            p = sum(cur_root.y) / len(cur_root.y)
            return {1: p, 0: 1-p}
        else:
            return self.predict_instance(cur_root.left if x[cur_root.split_dim] < cur_root.split_value else cur_root.right, x)
    
    def predict_proba(self, X):
        answer = []
        for x in X.values:
            answer.append(self.predict_instance(self.root, np.array(x)))
        return answer
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return [max(p.keys(), key=lambda k: p[k]) for p in proba]