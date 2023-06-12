import numpy as np
import Parse_data
from classifier import DecisionTree,DecisionTreeVisualizer


def accuracy_score(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    n_correct = sum([1 if y_t == y_p else 0 for y_t, y_p in zip(y_true, y_pred)])
    accuracy = n_correct / len(y_true)
    return accuracy

test_images = './Mnist/Mnist/test-images.idx3-ubyte'
test_labels = './Mnist/Mnist/test-labels.idx1-ubyte'
train_images = './Mnist/Mnist/train-images.idx3-ubyte'
train_labels = './Mnist/Mnist/train-labels.idx1-ubyte'

x_train = Parse_data.decode_idx3_ubyte(train_images)
x_test =  Parse_data.decode_idx3_ubyte(test_images)
y_train = Parse_data.decode_idx1_ubyte(train_labels)
y_test =  Parse_data.decode_idx1_ubyte(test_labels)

x_test = x_test.reshape(10000, 28*28)
x_train = x_train.reshape(60000, 28*28)
y_train = np.array(y_train)
y_test = np.array(y_test)

# 构建决策树分类器
tree = DecisionTree(max_depth=10, min_samples_split=2, criterion='gini')

# 训练决策树
tree.fit(x_train, y_train)
# tree_visualizer = DecisionTreeVisualizer(tree)
# tree_visualizer.visualize(class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],filename='tree')



# 在测试集上评估模型
y_pred = tree.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.3f}")
