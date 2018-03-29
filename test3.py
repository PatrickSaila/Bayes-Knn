# coding=utf-8
import numpy as np
import random

# conversion_labels = {'republican' : 0, 'democrat' : 1,
#                          'n' : 0, 'y' : 1, '?' : 2}
#
# data_congress = np.loadtxt('datasets/house-votes-84.data', delimiter=',',
#                       dtype={'names': ('col0', 'col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8'
#                                        , 'col9', 'col10', 'col11', 'col12', 'col13', 'col14', 'col15', 'col16'),
#                              'formats': ('|S10', '|S9', '|S9', '|S9', '|S9', '|S9', '|S9', '|S9', '|S9', '|S9', '|S9', '|S9', '|S9'
#                                          , '|S9', '|S9', '|S9', '|S9')})
#
# numerical_data_congress = np.array(data_congress).reshape(-1,).tolist()
# #print numerical_data_congress
#
# j = 0
# for i in numerical_data_congress:
#     i = [conversion_labels.get(item, item) for item in i]
#     numerical_data_congress[j] = i
#     j += 1
#
# print numerical_data_congress
# print type(numerical_data_congress[0][0])

# conversion_labels = {'Iris-setosa': 0, 'Iris-versicolor' : 1, 'Iris-virginica' : 2}
# data = np.loadtxt('datasets/bezdekIris.data', delimiter=',',dtype={'names': ('col0', 'col1', 'col2', 'col3', 'col4'),
#                         'formats': (float, float, float, float, '|S20')})
#
# data = np.array(data).reshape(-1, ).tolist()
#
# j = 0
# for i in data:
#     i = [conversion_labels.get(item, item) for item in i]
#     data[j] = i
#     j += 1
#
# data_setosa = []
# data_versicolor = []
# data_virginica = []
# for i in data:      # pour faire une sélection stratifiée
#     if i[4] == 0:
#         data_setosa.append(i)
#     elif i[4] == 1:
#         data_versicolor.append(i)
#         #print "Iris-versicolor: {}".format(i)
#     else:
#         data_virginica.append(i)
#
# print data_setosa

x = [101, 2, 0, 3, 3, 2, 0, 3, 2, "name"]

# print "{:>32}".format("Actual class")
# print "{:>19}    {}    {}".format("Cat", "Dog", "Rabbit")
# print "{}{:>5}{:>7}{:>9}".format("Predicted Cat", x[0], x[1], x[2])
# print "{}{:>5}{:>7}{:>9}".format("Predicted Dog", x[3], x[4], x[5])
# print "{}{:>2}{:>7}{:>9}".format("Predicted Rabbit", x[6], x[7], x[8])
x[0], x[-1] = x[-1], x[0]
x = x[1:]
print x


