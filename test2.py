# coding=utf-8
import numpy as np
import random

#data = np.genfromtxt('datasets/bezdekIris.data', delimiter=',', dtype=None, names=('sepal length', 'sepal width', 'petal length', 'petal width', 'label'))
data = np.loadtxt('datasets/bezdekIris.data', delimiter=',',
           dtype={'names': ('col0', 'col1', 'col2', 'col3', 'col4'),
                  'formats': (float, float, float, float, '|S20')})
#print data
data_setosa = []
data_versicolor = []
data_virginica = []
for i in data:
    if i[4] == 'Iris-setosa':
        data_setosa.append(i)
        print "Iris-setosa: {}".format(i)
    elif i[4] == 'Iris-versicolor':
        data_versicolor.append(i)
        print "Iris-versicolor: {}".format(i)
    else:
        data_virginica.append(i)
        print "Iris-virginica: {}".format(i)

print "lenght de toute les data: {}".format(len(data))
print "lenght de toute les setosa: {}".format(len(data_setosa))
print "lenght de toute les versicolor: {}".format(len(data_versicolor))
print "lenght de toute les virginica: {}".format(len(data_virginica))

nombre_pour_entrainement = int((len(data) * 0.60) / 3)
print "nombre_pour_entrainement: {}".format(nombre_pour_entrainement)

train = []
train_labels = []
test = []
test_labels = []

np.random.permutation(data_setosa)
for i in range(0, nombre_pour_entrainement):
    train.append(data_setosa[i])
    train_labels.append(data_setosa[i][4])
for i in range(nombre_pour_entrainement, 50):
    test.append(data_setosa[i])
    test_labels.append(data_setosa[i][4])
np.random.permutation(data_versicolor)
for i in range(0, nombre_pour_entrainement):
    train.append(data_versicolor[i])
    train_labels.append(data_versicolor[i][4])
for i in range(nombre_pour_entrainement, 50):
    test.append(data_versicolor[i])
    test_labels.append(data_versicolor[i][4])
np.random.permutation(data_virginica)
for i in range(0, nombre_pour_entrainement):
    train.append(data_virginica[i])
    train_labels.append(data_virginica[i][4])
for i in range(nombre_pour_entrainement, 50):
    test.append(data_virginica[i])
    test_labels.append(data_virginica[i][4])

print len(train)
print len(test)
print len(train_labels)
print len(test_labels)

