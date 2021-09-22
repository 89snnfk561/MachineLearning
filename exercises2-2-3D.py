import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB

# init
nb_samples = 50
X, Y = make_classification(n_samples=nb_samples, n_features=3, n_clusters_per_class=1, n_classes=3, n_informative=2, n_redundant=0)

# 畫資料點
def show_dataset(X, Y):
    # fig, ax = plt.subplots(1, 1, figsize=(30, 25))

    # 3D
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.grid()
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')

    for i in range(nb_samples):
        if Y[i] == 0:
            ax.scatter(X[i, 0], X[i, 1], X[i, 2], marker='o', color='r', s = 40)
        elif Y[i] == 1:
            ax.scatter(X[i, 0], X[i, 1], X[i, 2], marker='^', color='b', s = 40)
        else:
            ax.scatter(X[i, 0], X[i, 1], X[i, 2], marker='^', color='g', s = 40)
    plt.tick_params(axis='both', labelsize=20, color='blue')
    plt.show()


show_dataset(X, Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

bnb = BernoulliNB(binarize=0.0) #化作binary
bnb.fit(X_train, Y_train)
print('Bernoulli Naive Bayes score: %.3f' % bnb.score(X_test, Y_test))
bnb_scores = cross_val_score(bnb, X, Y, scoring='accuracy', cv=10)
print('Bernoulli Naive Bayes CV average score: %.3f' % bnb_scores.mean())

data = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
Yp1 = bnb.predict(data)
print(Yp1)
print()

gnb = GaussianNB()
gnb.fit(X_train, Y_train)
print('GaussianNB Naive Bayes score: %.3f' % gnb.score(X_test, Y_test))
gnb_scores = cross_val_score(gnb, X, Y, scoring='accuracy', cv=10)
print('GaussianNB Naive Bayes CV average score: %.3f' % gnb_scores.mean())

# Predict some values
Yp2 = gnb.predict(data)
print(Yp2)
