# coding=utf-8
import numpy as np
import math

"""
Vous allez definir une classe pour chaque algorithme que vous allez développer,
votre classe doit contenit au moins les 3 methodes definies ici bas,
    * train : pour entrainer le modèle sur l'ensemble d'entrainement
    * predict   : pour prédire la classe d'un exemple donné
    * test      : pour tester sur l'ensemble de test
vous pouvez rajouter d'autres méthodes qui peuvent vous etre utiles, mais moi
je vais avoir besoin de tester les méthodes test, predict et test de votre code.
"""


class BayesNaif():

    def __init__(self, p_dataset): #Je peux passer d'autres param au besoin
        self.dataset = p_dataset

    def moyenneEtSigma(self, p_data):   # Moyenne et écart type pour nos probabilités gaussiennes
        moyenne = sum(p_data) / float(len(p_data))
        sigma = math.sqrt(sum([pow(x - moyenne,2) for x in p_data]) / float(len(p_data)-1)) # len -1 puisque j'ai gardé les labels
        return (moyenne, sigma)

    def decoupageClasse(self, p_train):     # découpe le dataset par classes
        data_trier = {}
        for i in range(len(p_train)):
            temp = p_train[i]
            if (temp[-1] not in data_trier):
                data_trier[temp[-1]] = []
            data_trier[temp[-1]].append(temp)
        return data_trier

    def moyEtSigClasse(self, dataset):
        decouper = self.decoupageClasse(dataset)
        total = {}
        for i, j in decouper.iteritems():
            par_attribut = [(self.moyenneEtSigma(k)[0], self.moyenneEtSigma(k)[1]) for k in zip(*j)]
            del par_attribut[-1:]  # pour mes labels
            total[i] = par_attribut
        return total

    def probabilite(self, p_att, p_moy, p_sigma):
        if p_sigma == 0:
            p_sigma = 0.01
        rep = math.exp(-(math.pow(p_att-p_moy,2)/(2*math.pow(p_sigma,2))))
        proba = (1 / (math.sqrt(2*math.pi) * p_sigma)) * rep
        return proba

    def probaTotales(self, p_totales, p_test):
        proba = {}
        for i, j in p_totales.iteritems():
            proba[i] = 1
            for k in range(len(j)):
                moyenne, sigma = j[k]
                x = p_test[k]
                proba[i] *= self.probabilite(x, moyenne, sigma)
        return proba

    def predictionFinale(self, p_totales, p_test):
        proba_totales = self.probaTotales(p_totales, p_test)
        classe_probable, proba = 2, -1  # 0 <= probabilité <= 1
        for i, j in proba_totales.iteritems():
            if classe_probable == 2 or j > proba:
                proba = j
                classe_probable = i
        return classe_probable

    def seriePredictions(self, p_totales, p_testSet):
        predictions = []
        for i in range(len(p_testSet)):
            result = self.predictionFinale(p_totales, p_testSet[i])
            predictions.append(result)
        return predictions

    def accuracyPrecisionRecallCM(self, testSet, predictions):
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
            for i in range(len(predictions)):   # On regarde combien de nos prédictions étaient pour chaque classe
                if predictions[i] == 0:
                    nbr_prediction_class_zero += 1
                elif predictions[i] == 1:
                    nbr_prediction_class_one += 1
                else:
                    nbr_prediction_class_two += 1

            for j in range(len(testSet)):   # On regarde combien d'instance de chaque classe il y avait, et le nombre de bonnes prédictions
                if testSet[j][-1] == 0:
                    instance_of_class_zero += 1
                    if testSet[j][-1] == predictions[j]:
                        correct_class_zero += 1
                    elif predictions[j] == 1:
                        predicted_class_one_was_zero += 1
                    else:   # p_predictions[i] == 2
                        predicted_class_two_was_zero += 1
                elif testSet[j][-1] == 1:
                    instance_of_class_one += 1
                    if testSet[j][-1] == predictions[j]:
                        correct_class_one += 1
                    elif predictions[j] == 0:
                        predicted_class_zero_was_one += 1
                    else:   # p_predictions[i] == 2
                        predicted_class_two_was_one += 1
                else:
                    instance_of_class_two += 1
                    if testSet[j][-1] == predictions[j]:
                        correct_class_two += 1
                    elif predictions[j] == 0:
                        predicted_class_zero_was_two += 1
                    else:   # p_predictions[i] == 1
                        predicted_class_one_was_two += 1

            recall_zero = (correct_class_zero / float(instance_of_class_zero)) * 100.0
            recall_one = (correct_class_one / float(instance_of_class_one)) * 100.0
            recall_two = (correct_class_two / float(instance_of_class_two)) * 100.0
            precision_zero = (correct_class_zero / float(nbr_prediction_class_zero)) * 100.0
            precision_one = (correct_class_one / float(nbr_prediction_class_one)) * 100.0
            precision_two = (correct_class_two / float(nbr_prediction_class_two)) * 100.0
            accuracy_total = ((correct_class_zero + correct_class_one + correct_class_two)/float(len(testSet))) * 100.0
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

        elif self.dataset == "congres dataset":
            nbr_prediction_class_republican = 0
            nbr_prediction_class_democrate = 0
            for i in range(len(predictions)):
                if predictions[i] == 0:
                    nbr_prediction_class_republican += 1
                else:
                    nbr_prediction_class_democrate += 1

            correct_class_republican = 0
            correct_class_democrate = 0
            nbr_instances_class_republican = 0
            nbr_instances_class_democrate = 0
            for j in range(len(testSet)):
                if testSet[j][-1] == 0:
                    nbr_instances_class_republican += 1
                    if testSet[j][-1] == predictions[j]:
                        correct_class_republican +=1
                else:
                    nbr_instances_class_democrate += 1
                    if testSet[j][-1] == predictions[j]:
                        correct_class_democrate += 1

            accuracy_and_recall = []
            precision_republican = (correct_class_republican / float(
                correct_class_republican + (nbr_instances_class_democrate - correct_class_democrate))) * 100.0
            precision_democrat = (correct_class_democrate / float(
                correct_class_democrate + (nbr_instances_class_republican - correct_class_republican))) * 100.0
            recall_republican = (correct_class_republican / float(nbr_instances_class_republican)) * 100.0
            recall_democrat = (correct_class_democrate / float(nbr_instances_class_democrate)) * 100.0
            accuracy_total = ((correct_class_republican + correct_class_democrate) / float(len(testSet))) * 100
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
            for i in range(len(predictions)):
                if predictions[i] == 0:
                    nbr_prediction_class_zero += 1
                else:
                    nbr_prediction_class_one += 1

            correct_class_zero = 0
            correct_class_one = 0
            nbr_instances_class_zero = 0
            nbr_instances_class_one = 0
            for j in range(len(testSet)):
                if testSet[j][-1] == 0:
                    nbr_instances_class_zero += 1
                    if testSet[j][-1] == predictions[j]:
                        correct_class_zero += 1
                else:
                    nbr_instances_class_one += 1
                    if testSet[j][-1] == predictions[j]:
                        correct_class_one += 1

            accuracy_and_recall = []
            precision_zero = (correct_class_zero / float(nbr_prediction_class_zero)) * 100.0
            precision_one = (correct_class_one / float(nbr_prediction_class_one)) * 100.0
            recall_zero = (correct_class_zero / float(nbr_instances_class_zero)) * 100.0
            recall_one = (correct_class_one / float(nbr_instances_class_one)) * 100.0
            accuracy_total = ((correct_class_zero + correct_class_one) / float(len(testSet))) * 100
            confusion_matrix = []
            confusion_matrix.append(correct_class_zero)
            confusion_matrix.append(correct_class_one)
            confusion_matrix.append(nbr_instances_class_zero)
            confusion_matrix.append(nbr_instances_class_one)
            accuracy_and_recall.append(precision_zero)
            accuracy_and_recall.append(precision_one)
            accuracy_and_recall.append(recall_zero)
            accuracy_and_recall.append(recall_one)
            accuracy_and_recall.append(accuracy_total)
            accuracy_and_recall.append(confusion_matrix)

            return accuracy_and_recall

# =============================================================================

    def train(self, p_train, p_train_labels, p_dataset_name):
        if self.dataset == "iris dataset":
            trainingSet = p_train
            totales_par_classes = self.moyEtSigClasse(p_train)
            prediction = self.seriePredictions(totales_par_classes, trainingSet)
            print "Pour le {}, en mode bayes naïf:\n".format(p_dataset_name)
            accuracy = self.accuracyPrecisionRecallCM(trainingSet, prediction)
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
            trainingSet = p_train
            totales_par_classes = self.moyEtSigClasse(p_train)
            prediction = self.seriePredictions(totales_par_classes, trainingSet)
            accuracy = self.accuracyPrecisionRecallCM(trainingSet, prediction)
            print "Pour le {}, en mode bayes naïf:\n".format(p_dataset_name)
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

        else:   # Pour monks
            trainingSet = p_train
            totales_par_classes = self.moyEtSigClasse(p_train)
            prediction = self.seriePredictions(totales_par_classes, trainingSet)
            accuracy = self.accuracyPrecisionRecallCM(trainingSet, prediction)
            print "Pour le {}, en mode bayes naïf:\n".format(p_dataset_name)
            print "La precision et le recall sur le training_set est de: {}% et {}% pour un monk de classe 0".format(
                accuracy[0], accuracy[2])
            print "La precision et le recall sur le training_set est de: {}% et {}% pour un monk de classe 1".format(
                accuracy[1], accuracy[3])
            print "L'accuracy sur le training_set est de: {}%".format(accuracy[4])
            print "Voici la matrice de confusion du training_set:\n"

            cm = accuracy[5]
            print "{:>37}".format("Actual class")
            print "{:>29}    {}".format("monk 0", "monk 1")
            print "{}{:>10}{:>12}".format("Predicted monk 0", cm[0], cm[3] - cm[1])
            print "{}{:>10}{:>12}\n\n".format("Predicted monk 1", cm[2] - cm[0], cm[1])

    def test(self, p_test, test_labels, p_dataset_name):
        if self.dataset == "iris dataset":
            totales_par_classes = self.moyEtSigClasse(p_test)
            prediction = self.seriePredictions(totales_par_classes, p_test)
            print "Pour le {}, en mode bayes naïf:\n".format(p_dataset_name)
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
            trainingSet = p_test
            totales_par_classes = self.moyEtSigClasse(p_test)
            prediction = self.seriePredictions(totales_par_classes, trainingSet)
            accuracy = self.accuracyPrecisionRecallCM(trainingSet, prediction)
            print "Pour le {}, en mode bayes naïf:\n".format(p_dataset_name)
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
            trainingSet = p_test
            totales_par_classes = self.moyEtSigClasse(p_test)
            prediction = self.seriePredictions(totales_par_classes, trainingSet)
            accuracy = self.accuracyPrecisionRecallCM(trainingSet, prediction)
            print "Pour le {}, en mode bayes naïf:\n".format(p_dataset_name)
            print "La precision et le recall sur le test_set est de: {}% et {}% pour un monk de classe 0".format(
                accuracy[0], accuracy[2])
            print "La precision et le recall sur le test_set est de: {}% et {}% pour un monk de classe 1".format(
                accuracy[1], accuracy[3])
            print "L'accuracy sur le test_set est de: {}%".format(accuracy[4])
            print "Voici la matrice de confusion du test_set:\n"

            cm = accuracy[5]
            print "{:>37}".format("Actual class")
            print "{:>29}   {}".format("monk 0", "monk 1")
            print "{}{:>10}{:>12}".format("Predicted monk 0", cm[0], cm[3] - cm[1])
            print "{}{:>10}{:>12}\n\n".format("Predicted monk 1", cm[2] - cm[0], cm[1])
