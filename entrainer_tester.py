# coding=utf-8
import load_datasets
import BayesNaif # importer la classe du classifieur bayesien
import Knn # importer la classe du Knn
import numpy as np
import time
#importer d'autres fichiers et classes si vous en avez développés

"""
C'est le fichier main duquel nous allons tout lancer
Vous allez dire en commentaire c'est quoi les paramètres que vous avez utilisés
En gros, vous allez :
1- Initialiser votre classifieur avec ses paramètres
2- Charger les datasets
3- Entrainer votre classifieur
4- Le tester

"""

# On initialise les classifieurs BayesNaïf en leur indiquant le dataset
iris_bayes = BayesNaif.BayesNaif("iris dataset")
congressional_bayes = BayesNaif.BayesNaif("congres dataset")
monks_bayes = BayesNaif.BayesNaif("monks dataset")
# On initialise les classifieurs Knn en leur indiquant le dataset
iris_knn = Knn.Knn("iris dataset")
congressional_knn = Knn.Knn("congres dataset")
monks_knn = Knn.Knn("monks dataset")

# On charge les 4 dataset, et pour chacun on les sépare en 4 np_matrix
iris_train_dataset, iris_train_labels, iris_test_dataset, iris_test_labels =\
    load_datasets.load_iris_dataset(0.60)   # On utilise un ratio de 0.60 pour les instances qui vont servir à l'entrainement
congressional_train_dataset, congressional_train_labels, congressional_test_dataset, congressional_test_labels =\
    load_datasets.load_congressional_dataset(0.60)  # On utilise un ratio de 0.60 pour les instances qui vont servir à l'entrainement
monks_train_dataset, monks_train_labels, monks_test_dataset, monks_test_labels =\
    load_datasets.load_monks_dataset(2) # Ici on utilise les sets numéro 2

# On entraine nos classifieurs et puis on fait les tests
iris_bayes.train(iris_train_dataset, iris_train_labels, "iris dataset")
iris_bayes.test(iris_test_dataset, iris_test_labels, "iris dataset")
iris_knn.train(iris_train_dataset, iris_train_labels, "iris dataset")
iris_knn.test(iris_test_dataset, iris_test_labels, "iris dataset")

congressional_bayes.train(congressional_train_dataset, congressional_train_labels, "congressional dataset")
congressional_bayes.test(congressional_test_dataset, congressional_test_labels, "congressional dataset")
congressional_knn.train(congressional_train_dataset, congressional_train_labels, "congres dataset")
congressional_knn.test(congressional_test_dataset, congressional_test_labels, "congressional dataset")

monks_bayes.train(monks_train_dataset, monks_train_labels, "monks dataset")
monks_bayes.test(monks_test_dataset, monks_test_labels, "monks dataset")
monks_knn.train(monks_train_dataset, monks_train_labels, "monks dataset")
monks_knn.test(monks_test_dataset, monks_test_labels, "monks dataset")
