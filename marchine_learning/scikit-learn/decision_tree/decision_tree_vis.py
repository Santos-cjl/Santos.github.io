from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load the iris dataset
iris = load_iris()
x,y = iris.data, iris.target

# Train a Decision Tree Classifier
clf = DecisionTreeClassifier( random_state=42)
clf.fit(x, y)

# Plot the decision tree
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.savefig('../../images/decision_tree_vis.png',dpi=1200)
plt.show()
