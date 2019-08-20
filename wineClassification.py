import numpy as np
import pandas as pd
# for graph
import matplotlib.pyplot as plt
import seaborn as sns
# for KNN model
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

# load datasets
wine = datasets.load_wine()
x = wine.data
y = wine.target

# standardization of datasets
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size = 0.3, random_state = 0, stratify = y)

# Cross Validation
neighbours = list(range(1, 50, 2))
cv_scores = []
for k in neighbours:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x_scaled, y, cv = 5)
    cv_scores.append(scores.mean())
mse = [1-x for x in cv_scores]

optimal_k = neighbours[mse.index(min(mse))]

# graph plot for Optimal Validation
plt.plot(neighbours, mse)
plt.xlabel('value of K')
plt.ylabel('Mean square error')
plt.show()

print('\n##### KNN model for optimal value of K =', optimal_k, '#####\n')
knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(x_train,y_train)
y_predict = knn.predict(x_test)
print('accuracy: ', metrics.accuracy_score(y_test, y_predict))