import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        print(len(X))
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # compute the distance
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
    
        # get the closest k
        k_indices = np.argsort(distances)[:self.k]
        print(k_indices)
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        print(k_nearest_labels)

        # majority voye
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
    
if __name__ == "__main__":
    cmap = ListedColormap(['#FF0000',"#00FF0D",'#0000FF'])

    iris = datasets.load_iris()
    # print(iris.target_names)
    # print(iris.feature_names)
    # print(iris.data.shape)
    X, y = iris.data, iris.target
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    plt.figure()
    plt.scatter(X[:,2],X[:,3], c=y, cmap=cmap, edgecolor='k', s=20)
    plt.show()

    clf = KNN(k=4) # 5
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    # print(predictions)

    acc = np.sum(predictions == y_test) / len(y_test)
    print(acc)

    plt.figure()
    plt.scatter(X_test[:,2], X_test[:,3], c=predictions, cmap=cmap, edgecolor='k', s=20)
    plt.title(f'KNN Predictions (k={clf.k})')
    plt.xlabel(iris.feature_names[2])
    plt.ylabel(iris.feature_names[3])
    plt.show()