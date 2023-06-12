import numpy as np
from graphviz import Digraph

class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None, is_leaf=False):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.is_leaf = is_leaf


def accuracy_score(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    n_correct = sum([1 if y_t == y_p else 0 for y_t, y_p in zip(y_true, y_pred)])
    print(n_correct)
    accuracy = n_correct / len(y_true)
    return accuracy

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=5,criterion='gini'):
        self.max_depth = max_depth #决策树的最大深度
        self.min_samples_split = min_samples_split #
        #表示在对决策树节点进行分裂时，至少要有多少个样本才能进行分裂，如果小于该值，则不再进行分裂,防止过拟合
        self.criterion = criterion
        self.tree = None
        
    def fit(self, X, y):
        self.n_classes = len(np.unique(y)) #标签中不重复的元素，即分类的个数
        self.n_features = X.reshape((X.shape[0], -1)).shape[1] #将X训练集中每一张图片28*28展开成一维的向量
        self.tree = self._grow_tree(X, y) #构建决策树
        print("训练完成")
    
    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _predict(self, inputs):
        node = self.tree
        while node.left:
            if inputs[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def _grow_tree(self, X, y, depth=0):
        '''
            递归构建决策树
        '''
        n_samples, n_features = X.shape # X.shape = (60000,784)
        n_labels = len(np.unique(y)) # n_labels = 10
        # 如果满足以下任一条件，则停止生长树
        # 1. 当前深度大于等于最大深度
        # 2. 样本中只有一种类别
        # 3. 样本数量小于最小划分样本数
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self._leaf_value(y) # 返回一个叶子节点
            return Node(value=leaf_value, is_leaf=True)

        best_threshold = None
        
        while best_threshold is None:
            
            # 随机选择 sqrt(n_features) 个特征,即从784列中随机选择28个不重复的列作为特征
            feature_idxs = np.random.choice(n_features, int(np.sqrt(n_features)), replace=False)
            # 根据最佳划分特征和最佳划分阈值划分样本
            best_feature_idx, best_threshold = self._best_criteria(X, y, feature_idxs)
            
        left_idxs, right_idxs = self._split(X[:, best_feature_idx], best_threshold) 
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feature_idx, best_threshold, left, right)
    
    def _best_criteria(self, X, y, feature_idxs, gain=0):
        '''
        给定特征集合(feature_idxs)中，找到最佳的分裂特征和分裂阈值
        '''
        best_gain = -np.inf                                #初始化信息增益为负无穷
        split_idx, split_threshold = None, None            #初始化分裂的特征索引和分裂的阈值为None
        for feature_idx in feature_idxs:                   #在特征集合中遍历每个特征
            thresholds = np.unique(X[:, feature_idx])      #获取当前特征列上的所有唯一值，作为可能的分裂阈值
            thresholds = thresholds[np.array([e is not None for e in thresholds])]
            for threshold in thresholds:                   #在当前特征的所有可能分裂阈值中遍历
                if threshold is not None:
                    left_idxs, right_idxs = self._split(X[:, feature_idx], threshold)
                    #通过阈值对数据进行分裂，获取左子树和右子树的样本索引
                    if len(left_idxs) == 0 or len(right_idxs) == 0:
                    #如果左子树或右子树为空，说明当前阈值无法实现分裂，直接跳过当前阈值
                        continue

                    gain = self._gini_gain(y, left_idxs, right_idxs) #计算当前分割的基尼增益
                    if gain > best_gain:                       #如果当前增益大于最佳增益，则将当前增益和分割特征、阈值更新为最佳值。
                        best_gain = gain
                        split_idx = feature_idx
                        split_threshold = threshold
        return split_idx, split_threshold

    def _split(self, X_column, threshold=np.inf):
        '''
        根据阈值将特征列X_column分割成左子集和右子集,并返回这些子集的索引。
        '''
        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column > threshold).flatten()
        return left_idxs, right_idxs
    
    def _gini_gain(self, y, left_idxs, right_idxs):
        '''
        计算基尼系数增益
        :param y: array, 标签数组
        :param left_idxs: array, 左子节点下标
        :param right_idxs: array, 右子节点下标
        :return: float, 基尼系数增益值
        '''
        p = len(left_idxs) / (len(left_idxs) + len(right_idxs))  # 计算左子节点在样本中的占比
        left_gini = self.gini(y[left_idxs])  # 计算左子节点的基尼系数
        right_gini = self.gini(y[right_idxs])  # 计算右子节点的基尼系数
        gain = self.gini(y) - p * left_gini - (1 - p) * right_gini  # 计算基尼系数增益值
        return gain
    
    
    def gini(self, y):
        '''
        计算标签的基尼系数
        '''
        _, counts = np.unique(y, return_counts=True)         #计算标签列表中不同标签值的数量
        probs = counts / len(y)                              #计算每一类的概率
        gini = 1 - np.sum(probs ** 2)
        return gini
    
    def _leaf_value(self, y):
        counts = np.bincount(y)
        return np.argmax(counts)
    
    def post_prune(self, X_val, y_val):
        """Post-prune the decision tree."""
        self.num_pruned = 0
        self._post_prune(X_val, y_val, self.tree)
        print(f"Pruned {self.num_pruned} nodes.")

    def _post_prune(self, X_val, y_val, node):
        if node.is_leaf:
            return

        # Recursively prune children
        if not node.left.is_leaf:
            self._post_prune(X_val, y_val, node.left)
        if not node.right.is_leaf:
            self._post_prune(X_val, y_val, node.right)

        # Calculate the error with and without the node's children
        y_val_pred = self.predict(X_val)
        error_without_prune = np.sum(y_val_pred != y_val) / len(y_val)
        node_left = node.left
        node_right = node.right
        node.left = None
        node.right = None
        y_val_pred = self.predict(X_val)
        error_with_prune = np.sum(y_val_pred != y_val) / len(y_val)

        # Decide whether to prune the node or not
        if error_with_prune <= error_without_prune:
            node.is_leaf = True
            node.left = None
            node.right = None
        else:
            node.left = node_left
            node.right = node_right

class DecisionTreeVisualizer:
    def __init__(self, tree):
        self.tree = tree
        
    def visualize(self, feature_names=None, class_names=None, filename='tree'):
        dot = Digraph(comment='Decision Tree')
        dot.node_attr.update(shape='ellipse')
        self._add_node(dot, self.tree, feature_names, class_names)
        dot.render(filename, view=True)
        
    def _add_node(self, dot, node, feature_names, class_names):
        if feature_names is None:
            feature_name = str(node.feature_idx)
        else:
            feature_name = feature_names[node.feature_idx]
        
        if node.value is not None:
            if class_names is None:
                class_name = str(node.value)
            else:
                class_name = class_names[node.value]
            dot.node(str(id(node)), label=class_name, shape='box')
        else:
            dot.node(str(id(node)), label=feature_name)
            self._add_node(dot, node.left, feature_names, class_names)
            self._add_node(dot, node.right, feature_names, class_names)
            dot.edge(str(id(node)), str(id(node.left)), '<=' + str(node.threshold))
            dot.edge(str(id(node)), str(id(node.right)), '>' + str(node.threshold))