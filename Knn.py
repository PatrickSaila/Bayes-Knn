# coding=utf-8
import numpy as np
import math
import operator


class Knn():

    def __init__(self, p_dataset): # Je peux passer d'autres param au besoin
        self.dataset = p_dataset

    def distanceEuclidienne(self, p_debut, p_arrivee, p_limite):
        distance = 0
        for i in range(p_limite):
            distance += pow((p_debut[i] - p_arrivee[i]), 2)
        return math.sqrt(distance)

    def trouverVoisins(self, p_trainingSet, p_initiateur, k):
        distances = []
        length_initiateur = len(p_initiateur) - 1
        for i in range(len(p_trainingSet)):
            distance = self.distanceEuclidienne(p_initiateur, p_trainingSet[i], length_initiateur)
            distances.append((p_trainingSet[i], distance))
        distances.sort(key=operator.itemgetter(1))
        voisins = []
        for j in range(k):
            voisins.append(distances[j][0])
        return voisins

    def proba(self, p_voisins):
        proba = {}
        for i in range(len(p_voisins)):
            voisin = p_voisins[i][-1]
            if voisin in proba:
                proba[voisin] += 1
            else:
                proba[voisin] = 1
        proba_done = sorted(proba.iteritems(), key=operator.itemgetter(1), reverse=True)
        return proba_done[0][0]

    def accuracyPrecisionRecallCM(self, p_testSet, p_predictions):
        if self.dataset == "iris dataset":
            correct_class_zero = 0
            instance_of_class_zero = 0
            correct_class_one = 0
            instance_of_class_one = 0
            correct_class_two = 0
            instance_of_class_two = 0

            nbr_prediction_class_zero = 0
            nbr_prediction_class_one = 0
            nbr_prediction_class_two = 0
            predicted_class_one_was_zero = 0
            predicted_class_two_was_zero = 0
            predicted_class_zero_was_one = 0
            predicted_class_two_was_one = 0
            predicted_class_zero_was_two = 0
            predicted_class_one_was_two = 0
            for i in range(len(p_predictions)):
                if p_predictions[i] == 0:
                    nbr_prediction_class_zero += 1
                elif p_predictions[i] == 1:
                    nbr_prediction_class_one += 1
                else:
                    nbr_prediction_class_two += 1

            for i in range(len(p_testSet)):
                if p_testSet[i][-1] == 0:
                    instance_of_class_zero += 1
                    if p_testSet[i][-1] == p_predictions[i]:
                        correct_class_zero += 1
                    elif p_predictions[i] == 1:
                        predicted_class_one_was_zero += 1
                    else:   # p_predictions[i] == 2
                        predicted_class_two_was_zero += 1

                elif p_testSet[i][-1] == 1:
                    instance_of_class_one += 1
                    if p_testSet[i][-1] == p_predictions[i]:
                        correct_class_one += 1
                    elif p_predictions[i] == 0:
                        predicted_class_zero_was_one += 1
                    else:   # p_predictions[i] == 2
                        predicted_class_two_was_one += 1

                else:
                    instance_of_class_two += 1
                    if p_testSet[i][-1] == p_predictions[i]:
                        correct_class_two += 1
                    elif p_predictions[i] == 0:
                        predicted_class_zero_was_two += 1
                    else:   # p_predictions[i] == 1
                        predicted_class_one_was_two += 1

            recall_zero = (correct_class_zero / float(instance_of_class_zero)) * 100.0
            recall_one = (correct_class_one / float(instance_of_class_one)) * 100.0
            recall_two = (correct_class_two / float(instance_of_class_two)) * 100.0
            precision_zero = (correct_class_zero / float(nbr_prediction_class_zero)) * 100.0
            precision_one = (correct_class_one / float(nbr_prediction_class_one)) * 100.0
            precision_two = (correct_class_two / float(nbr_prediction_class_two)) * 100.0
            accuracy_total = ((correct_class_zero + correct_class_one + correct_class_two)/float(len(p_testSet))) * 100.0
            confusion_matrix = []
            confusion_matrix.append(correct_class_zero)
            confusion_matrix.append(predicted_class_zero_was_one)
            confusion_matrix.append(predicted_class_zero_was_two)
            confusion_matrix.append(predicted_class_zero_was_one)
            confusion_matrix.append(correct_class_one)
            confusion_matrix.append(predicted_class_two_was_one)
            confusion_matrix.append(predicted_class_zero_was_two)
            confusion_matrix.append(predicted_class_one_was_two)
            confusion_matrix.append(correct_class_two)
            accuracy_and_recall = []
            accuracy_and_recall.append(precision_zero)
            accuracy_and_recall.append(precision_one)
            accuracy_and_recall.append(precision_two)
            accuracy_and_recall.append(recall_zero)
            accuracy_and_recall.append(recall_one)
            accuracy_and_recall.append(recall_two)
            accuracy_and_recall.append(confusion_matrix)
            accuracy_and_recall.append(accuracy_total)

            return accuracy_and_recall
        elif self.dataset == "congress dataset":
            nbr_prediction_class_republican = 0
            nbr_prediction_class_democrate = 0
            for i in range(len(p_predictions)):
                if p_predictions[i] == 0:
                    nbr_prediction_class_republican += 1
                else:
                    nbr_prediction_class_democrate += 1

            correct_class_republican = 0
            correct_class_democrate = 0
            nbr_instances_class_republican = 0
            nbr_instances_class_democrate = 0
            for x in range(len(p_testSet)):
                if p_testSet[x][-1] == 0:
                    nbr_instances_class_republican += 1
                    if p_testSet[x][-1] == p_predictions[x]:
                        correct_class_republican +=1
                else:
                    nbr_instances_class_democrate += 1
                    if p_testSet[x][-1] == p_predictions[x]:
                        correct_class_democrate += 1


            accuracy_and_recall = []
            precision_republican = (correct_class_republican / float(correct_class_republican + (nbr_instances_class_democrate - correct_class_democrate))) * 100.0
            precision_democrat = (correct_class_democrate / float(correct_class_democrate + (nbr_instances_class_republican - correct_class_republican))) * 100.0
            recall_republican = (correct_class_republican / float(nbr_instances_class_republican)) * 100.0
            recall_democrat = (correct_class_democrate / float(nbr_instances_class_democrate)) * 100.0
            accuracy_total = ((correct_class_republican + correct_class_democrate) / float (len(p_testSet))) * 100
            accuracy_and_recall.append(precision_republican)
            accuracy_and_recall.append(precision_democrat)
            accuracy_and_recall.append(recall_republican)
            accuracy_and_recall.append(recall_democrat)
            accuracy_and_recall.append(accuracy_total)
            confusion_matrix = []
            confusion_matrix.append(correct_class_republican)
            confusion_matrix.append(correct_class_democrate)
            confusion_matrix.append(nbr_instances_class_republican)
            confusion_matrix.append(nbr_instances_class_democrate)
            accuracy_and_recall.append(confusion_matrix)

            return accuracy_and_recall

        else:   # monks
            nbr_prediction_class_zero = 0
            nbr_prediction_class_one = 0
            for i in range(len(p_predictions)):
                if p_predictions[i] == 0:
                    nbr_prediction_class_zero += 1
                else:
                    nbr_prediction_class_one += 1

            correct_class_zero = 0
            correct_class_one = 0
            nbr_instances_class_zero = 0
            nbr_instances_class_one = 0
            for x in range(len(p_testSet)):
                if p_testSet[x][-1] == 0:
                    nbr_instances_class_zero += 1
                    if p_testSet[x][-1] == p_predictions[x]:
                        correct_class_zero += 1
                else:
                    nbr_instances_class_one += 1
                    if p_testSet[x][-1] == p_predictions[x]:
                        correct_class_one += 1

            accuracy_and_recall = []
            precision_zero = correct_class_zero / float((correct_class_zero + (nbr_instances_class_one - correct_class_one))) * 100.0
            precision_one = correct_class_one / float((correct_class_one + (nbr_instances_class_zero - correct_class_zero))) * 100.0
            recall_zero = (correct_class_zero / float(nbr_instances_class_zero)) * 100.0
            recall_one = (correct_class_one / float(nbr_instances_class_one)) * 100.0
            accuracy_total = ((correct_class_zero + correct_class_one) / float(len(p_testSet))) * 100
            accuracy_and_recall.append(precision_zero)
            accuracy_and_recall.append(precision_one)
            accuracy_and_recall.append(recall_zero)
            accuracy_and_recall.append(recall_one)
            accuracy_and_recall.append(accuracy_total)
            confusion_matrix = []
            confusion_matrix.append(correct_class_zero)
            confusion_matrix.append(correct_class_one)
            confusion_matrix.append(nbr_instances_class_zero)
            confusion_matrix.append(nbr_instances_class_one)
            accuracy_and_recall.append(confusion_matrix)

            return accuracy_and_recall

# =============================================================================

    def train(self, p_train, p_train_labels, p_dataset_name):
        prediction = []
        k = 5
        for i in range(len(p_train)):
            voisins = self.trouverVoisins(p_train, p_train[i], k)
            done = self.proba(voisins)
            prediction.append(done)

        if self.dataset == "iris dataset":
            print "Pour le {}, en mode Knn:\n".format(p_dataset_name)
            accuracy = self.accuracyPrecisionRecallCM(p_train, prediction)
            print "La precision sur le training_set est respectivement de: {}%, {}% et {}% pour les iris de type setosa, " \
                  "versicolor et virginica.".format(accuracy[3], accuracy[4], accuracy[5])
            print "Le rappel sur le training_set est respectivement de: {}%, {}% et {}%. pour les iris de type setosa, " \
                  "versicolor et virginica".format(accuracy[0], accuracy[1], accuracy[2])
            print "L'accuracy sur le training_set est de: {}%".format(accuracy[7])
            print "Voici la matrice de confusion du training_set:\n"

            print "{:>42}".format("Actual class")
            print "{:>29} {:>11} {:>9}".format("setosa", "versicolor", "virginica")
            print "{} {:>11} {:>8} {:>9}".format("Predicted setosa", accuracy[6][0], accuracy[6][1], accuracy[6][2])
            print "{} {:>6} {:>10} {:>8}".format("Predicted versicolor", accuracy[6][3], accuracy[6][4], accuracy[6][5])
            print "{} {:>7} {:>9} {:>9}\n".format("Predicted virginica", accuracy[6][6], accuracy[6][7], accuracy[6][8])
            print "\n\n"

        elif self.dataset == "congres dataset":
            accuracy = self.accuracyPrecisionRecallCM(p_train, prediction)
            print "Pour le {}, en mode Knn:\n".format(p_dataset_name)
            print "La precision et le recall sur le training_set est de: {}% et {}% pour un républicain".format(
                accuracy[0], accuracy[2])
            print "La precision et le recall sur le training_set est de: {}% et {}% pour un démocrate".format(
                accuracy[1], accuracy[3])
            print "L'accuracy sur le training_set est de: {}%".format(accuracy[4])
            print "Voici la matrice de confusion du training_set:\n"

            cm = accuracy[5]
            print "{:>40}".format("Actual class")
            print "{:>32}   {}".format("republican", "democrate")
            print "{}{:>7}{:>13}".format("Predicted republican", cm[0], cm[3] - cm[1])
            print "{}{:>8}{:>13}\n\n".format("Predicted democrate", cm[2] - cm[0], cm[1])

        else:  # Pour monks
            accuracy = self.accuracyPrecisionRecallCM(p_train, prediction)
            print "Pour le {}, en mode Knn:\n".format(p_dataset_name)
            print "La precision et le recall sur le training_set est de: {}% et {}% pour un monk zero".format(
                accuracy[0], accuracy[2])
            print "La precision et le recall sur le training_set est de: {}% et {}% pour un monk one".format(
                accuracy[1], accuracy[3])
            print "L'accuracy sur le training_set est de: {}%".format(accuracy[4])
            print "Voici la matrice de confusion du training_set:\n"

            cm = accuracy[5]
            print "{:>37}".format("Actual class")
            print "{:>29}   {}".format("monk 0", "monk 1")
            print "{}{:>10}{:>12}".format("Predicted monk 0", cm[0], cm[3] - cm[1])
            print "{}{:>10}{:>12}\n\n".format("Predicted monk 1", cm[2] - cm[0], cm[1])

    def test(self, p_test, p_test_labels, p_dataset_name):
        prediction = []
        k = 5
        for i in range(len(p_test)):
            voisins = self.trouverVoisins(p_test, p_test[i], k)
            done = self.proba(voisins)
            prediction.append(done)

        if self.dataset == "iris dataset":
            print "Pour le {}, en mode Knn:\n".format(p_dataset_name)
            accuracy = self.accuracyPrecisionRecallCM(p_test, prediction)
            print "La precision sur le test_set est respectivement de: {}%, {}% et {}% pour les iris de type setosa, " \
                  "versicolor et virginica.".format(accuracy[3], accuracy[4], accuracy[5])
            print "Le rappel sur le test_set est respectivement de: {}%, {}% et {}%. pour les iris de type setosa, " \
                  "versicolor et virginica".format(accuracy[0], accuracy[1], accuracy[2])
            print "L'accuracy sur le test_set est de: {}%".format(accuracy[7])
            print "Voici la matrice de confusion du test_set:\n"

            print "{:>42}".format("Actual class")
            print "{:>29} {:>11} {:>9}".format("setosa", "versicolor", "virginica")
            print "{} {:>11} {:>8} {:>9}".format("Predicted setosa", accuracy[6][0], accuracy[6][1], accuracy[6][2])
            print "{} {:>6} {:>10} {:>8}".format("Predicted versicolor", accuracy[6][3], accuracy[6][4], accuracy[6][5])
            print "{} {:>7} {:>9} {:>9}\n".format("Predicted virginica", accuracy[6][6], accuracy[6][7], accuracy[6][8])
            print "\n\n"

        elif self.dataset == "congres dataset":
            accuracy = self.accuracyPrecisionRecallCM(p_test, prediction)
            print "Pour le {}, en mode Knn:\n".format(p_dataset_name)
            print "La precision et le recall sur le test_set est de: {}% et {}% pour un républicain".format(
                accuracy[0], accuracy[2])
            print "La precision et le recall sur le test_set est de: {}% et {}% pour un démocrate".format(
                accuracy[1], accuracy[3])
            print "L'accuracy sur le test_set est de: {}%".format(accuracy[4])
            print "Voici la matrice de confusion du test_set:\n"

            cm = accuracy[5]
            print "{:>40}".format("Actual class")
            print "{:>32}   {}".format("republican", "democrate")
            print "{}{:>7}{:>13}".format("Predicted republican", cm[0], cm[3] - cm[1])
            print "{}{:>8}{:>13}\n\n".format("Predicted democrate", cm[2] - cm[0], cm[1])

        else:  # Pour monks
            accuracy = self.accuracyPrecisionRecallCM(p_test, prediction)
            print "Pour le {}, en mode Knn:\n".format(p_dataset_name)
            print "La precision et le recall sur le test_set est de: {}% et {}% pour un monk 0".format(
                accuracy[0], accuracy[2])
            print "La precision et le recall sur le test_set est de: {}% et {}% pour un monk 1".format(
                accuracy[1], accuracy[3])
            print "L'accuracy sur le test_set est de: {}%".format(accuracy[4])
            print "Voici la matrice de confusion du test_set:\n"

            cm = accuracy[5]
            print "{:>37}".format("Actual class")
            print "{:>29}   {}".format("monk 0", "monk 1")
            print "{}{:>10}{:>12}".format("Predicted monk 0", cm[0], cm[3] - cm[1])
            print "{}{:>10}{:>12}\n\n".format("Predicted monk 1", cm[2] - cm[0], cm[1])