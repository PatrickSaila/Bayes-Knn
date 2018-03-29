# coding=utf-8
import numpy as np
import random

def load_iris_dataset(train_ratio):
    """Cette fonction a pour but de lire le dataset Iris

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont etre attribués à l'entrainement,
        le rest des exemples va etre utilisé pour les tests.
        Par exemple : si le ratio est 50%, il y aura 50% des exemple (75 exemples) qui vont etre utilisé
        pour l'entrainement, et 50% (75 exemples) pour le test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy, train, train_labels, test, et test_labels
		
        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple (ou instance) d'entrainement.
		
        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est le label (ou l'etiquette) pour l'exemple train[i]
        
        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple (ou instance) de test.
		
        - test_labels : contient les labels (ou les étiquettes) pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est le label (ou l'etiquette) pour l'exemple test[i]
    """
    
    random.seed(1) # Pour avoir les meme nombres aléatoires à chaque initialisation.
    
    # Vous pouvez utiliser des valeurs numériques pour les différents types de classes, tel que :
    conversion_labels = {'Iris-setosa': 0, 'Iris-versicolor' : 1, 'Iris-virginica' : 2}
    
    # Le fichier du dataset est dans le dossier datasets en attaché 
    #f = open('datasets/bezdekIris.data', 'r')
    
    
    # TODO : le code ici pour lire le dataset
    
    # REMARQUE très importante : 
	# remarquez bien comment les exemples sont ordonnés dans 
    # le fichier du dataset, ils sont ordonnés par type de fleur, cela veut dire que 
    # si vous lisez les exemples dans cet ordre et que si par exemple votre ration est de 60%,
    # vous n'allez avoir aucun exemple du type Iris-virginica pour l'entrainement, pensez
    # donc à utiliser la fonction random.shuffle pour melanger les exemples du dataset avant de séparer
    # en train et test.

    # ================== PAT: MY CODE ============
    data = np.loadtxt('datasets/bezdekIris.data', delimiter=',',dtype={'names': ('col0', 'col1', 'col2', 'col3', 'col4'),
                         'formats': (float, float, float, float, '|S20')})
    data = np.array(data).reshape(-1, ).tolist()    # met la numpy_matrix en liste pour des manipulations

    j = 0
    for i in data:  # change les types de classes en valeurs numériques
        i = [conversion_labels.get(k, k) for k in i]
        data[j] = i
        j += 1

    data_setosa = []
    data_versicolor = []
    data_virginica = []
    for i in data:      # pour faire une sélection stratifiée
        if i[4] == 0:
            data_setosa.append(i)
        elif i[4] == 1:
            data_versicolor.append(i)
        else:
            data_virginica.append(i)

    nombre_pour_entrainement = int((len(data) * train_ratio) / 3)

    train_list = []
    train_labels_list = []
    test_list = []
    test_labels_list = []

    np.random.permutation(data_setosa)
    for i in range(0, nombre_pour_entrainement):
        train_list.append(data_setosa[i])
        train_labels_list.append(data_setosa[i][4])
    for i in range(nombre_pour_entrainement, len(data_setosa)): # le reste va dans le test_set
        test_list.append(data_setosa[i])
        test_labels_list.append(data_setosa[i][4])
    np.random.permutation(data_versicolor)
    for i in range(0, nombre_pour_entrainement):
        train_list.append(data_versicolor[i])
        train_labels_list.append(data_versicolor[i][4])
    for i in range(nombre_pour_entrainement, len(data_versicolor)):
        test_list.append(data_versicolor[i])
        test_labels_list.append(data_versicolor[i][4])
    np.random.permutation(data_virginica)
    for i in range(0, nombre_pour_entrainement):
        train_list.append(data_virginica[i])
        train_labels_list.append(data_virginica[i][4])
    for i in range(nombre_pour_entrainement, len(data_virginica)):
        test_list.append(data_virginica[i])
        test_labels_list.append(data_virginica[i][4])

    # Tres important : la fonction doit retourner 4 matrices (ou vecteurs) de type Numpy.
    train = np.asarray(train_list)
    train_labels = np.asarray(train_labels_list)
    test = np.asarray(test_list)
    test_labels = np.asarray(test_labels_list)

    return (train, train_labels, test, test_labels)


def load_congressional_dataset(train_ratio):
    """Cette fonction a pour but de lire le dataset Congressional Voting Records

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont servir pour l'entrainement,
        le rest des exemples va etre utilisé pour les test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy, train, train_labels, test, et test_labels
        
        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple (ou instance) d'entrainement.
		
        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est le label (ou l'etiquette) pour l'exemple train[i]
        
        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple (ou instance) de test.
		
        - test_labels : contient les labels (ou les étiquettes) pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est le label (ou l'etiquette) pour l'exemple test[i]
    """
    
    random.seed(1) # Pour avoir les meme nombres aléatoires à chaque initialisation.
    
    # Vous pouvez utiliser un dictionnaire pour convertir les attributs en numériques 
    # Notez bien qu'on a traduit le symbole "?" pour une valeur numérique
    # Vous pouvez biensur utiliser d'autres valeurs pour ces attributs
    conversion_labels = {'republican' : 0, 'democrat' : 1, 
                         'n' : 0, 'y' : 1, '?' : 3}
    
    # Le fichier du dataset est dans le dossier datasets en attaché 
    #f = open('datasets/house-votes-84.data', 'r')

	
    # TODO : le code ici pour lire le dataset
    data_congres = np.loadtxt('datasets/house-votes-84.data', delimiter=',',
                              dtype={'names': ('col0', 'col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8'
                                               , 'col9', 'col10', 'col11', 'col12', 'col13', 'col14', 'col15', 'col16'),
                                     'formats': ('|S10', '|S9', '|S9', '|S9', '|S9', '|S9', '|S9', '|S9', '|S9', '|S9',
                                                 '|S9', '|S9','|S9', '|S9', '|S9', '|S9', '|S9')})
    data_congres = np.array(data_congres).reshape(-1, ).tolist()    # change la numpy_matrix en liste

    j = 0
    for i in data_congres:  # change les attributs qui étaient en strings en valeurs numériques
        i = [conversion_labels.get(item, item) for item in i]
        data_congres[j] = i
        j += 1

    np.random.permutation(data_congres)
    nombre_pour_entrainement = int((len(data_congres) * train_ratio))

    train_list = []
    train_labels_list = []
    test_list = []
    test_labels_list = []

    for i in range(0, nombre_pour_entrainement):    # la partie réservée pour l'entraînement
        train_list.append(data_congres[i])
        train_labels_list.append(data_congres[i][4])
    for i in range(nombre_pour_entrainement, len(data_congres)):    # pour les tests
        test_list.append(data_congres[i])
        test_labels_list.append(data_congres[i][4])

    # Tres important : la fonction doit retourner 4 matrices (ou vecteurs) de type Numpy.
    train = np.asarray(train_list)
    train_labels = np.asarray(train_labels_list)
    test = np.asarray(test_list)
    test_labels = np.asarray(test_labels_list)

    return (train, train_labels, test, test_labels)


def load_monks_dataset(numero_dataset):
    """Cette fonction a pour but de lire le dataset Monks
    
    Notez bien que ce dataset est différent des autres d'un point de vue
    exemples entrainement et exemples de tests.
    Pour ce dataset, nous avons 3 différents sous problèmes, et pour chacun
    nous disposons d'un fichier contenant les exemples d'entrainement et 
    d'un fichier contenant les fichiers de tests. Donc nous avons besoin 
    seulement du numéro du sous problème pour charger le dataset.

    Args:
        numero_dataset: lequel des sous problèmes nous voulons charger (1, 2 ou 3 ?)
		par exemple, si numero_dataset=2, vous devez lire :
			le fichier monks-2.train contenant les exemples pour l'entrainement
			et le fichier monks-2.test contenant les exemples pour le test
        les fichiers sont tous dans le dossier datasets
    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy, train, train_labels, test, et test_labels
        
        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple (ou instance) d'entrainement.
        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est le label (ou l'etiquette) pour l'exemple train[i]
        
        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple (ou instance) de test.
        - test_labels : contient les labels (ou les étiquettes) pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est le label (ou l'etiquette) pour l'exemple test[i]
    """

    # TODO : votre code ici, vous devez lire les fichiers .train et .test selon l'argument numero_dataset
    if numero_dataset == 1:
        monk_train_list = np.loadtxt('datasets/monks-1.train',
                               dtype = {'names': ('col0', 'col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7'),
                                        'formats': (float, float, float, float, float, float, float, '|S30')})

        monk_train_list = np.array(monk_train_list).reshape(-1, ).tolist()
        bean = 0
        for i in monk_train_list:   # pour placer les class en dernier et se débarasser des id individuels
            i = list(i)
            i[0], i[-1] = i[-1], i[0]
            i = i[1:]
            i = tuple(i)
            monk_train_list[bean] = i
            bean += 1

        train_labels_list = []
        for i in range(0, len(train_labels_list)):
            train_labels_list.append(monk_train_list[i][6])

        monk_test_list = np.loadtxt('datasets/monks-1.test',
                                     dtype={'names': ('col0', 'col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7'),
                                            'formats': (float, float, float, float, float, float, float, '|S30')})

        monk_test_list = np.array(monk_test_list).reshape(-1, ).tolist()
        bean = 0
        for i in monk_test_list:
            i = list(i)
            i[0], i[-1] = i[-1], i[0]
            i = i[1:]
            i = tuple(i)
            monk_test_list[bean] = i
            bean += 1

        test_labels_list = []
        for i in range(0, len(monk_test_list)):
            test_labels_list.append(monk_test_list[i][6])

    elif numero_dataset == 2:
        monk_train_list = np.loadtxt('datasets/monks-2.train',
                                     dtype={'names': ('col0', 'col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7'),
                                            'formats': (float, float, float, float, float, float, float, '|S30')})

        monk_train_list = np.array(monk_train_list).reshape(-1, ).tolist()
        bean = 0
        for i in monk_train_list:
            i = list(i)
            i[0], i[-1] = i[-1], i[0]
            i = i[1:]
            i = tuple(i)
            monk_train_list[bean] = i
            bean += 1

        train_labels_list = []
        for i in range(0, len(train_labels_list)):
            train_labels_list.append(monk_train_list[i][6])

        monk_test_list = np.loadtxt('datasets/monks-2.test',
                                    dtype={'names': ('col0', 'col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7'),
                                           'formats': (float, float, float, float, float, float, float, '|S30')})

        monk_test_list = np.array(monk_test_list).reshape(-1, ).tolist()
        bean = 0
        for i in monk_test_list:
            i = list(i)
            i[0], i[-1] = i[-1], i[0]
            i = i[1:]
            i = tuple(i)
            monk_test_list[bean] = i
            bean += 1

        test_labels_list = []
        for i in range(0, len(monk_test_list)):
            test_labels_list.append(monk_test_list[i][6])

    else:   # les dataset numéro 3
        monk_train_list = np.loadtxt('datasets/monks-3.train',
                                     dtype={'names': ('col0', 'col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7'),
                                            'formats': (float, float, float, float, float, float, float, '|S30')})

        monk_train_list = np.array(monk_train_list).reshape(-1, ).tolist()
        bean = 0
        for i in monk_train_list:
            i = list(i)
            i[0], i[-1] = i[-1], i[0]
            i = i[1:]
            i = tuple(i)
            monk_train_list[bean] = i
            bean += 1

        train_labels_list = []
        for i in range(0, len(train_labels_list)):
            train_labels_list.append(monk_train_list[i][6])

        monk_test_list = np.loadtxt('datasets/monks-3.test',
                                    dtype={'names': ('col0', 'col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7'),
                                           'formats': (float, float, float, float, float, float, float, '|S30')})

        monk_test_list = np.array(monk_test_list).reshape(-1, ).tolist()
        bean = 0
        for i in monk_test_list:
            i = list(i)
            i[0], i[-1] = i[-1], i[0]
            i = i[1:]
            i = tuple(i)
            monk_test_list[bean] = i
            bean += 1

        test_labels_list = []
        for i in range(0, len(monk_test_list)):
            test_labels_list.append(monk_test_list[i][6])

    # La fonction doit retourner 4 matrices (ou vecteurs) de type Numpy.
    train = np.asarray(monk_train_list)
    train_labels = np.asarray(train_labels_list)
    test = np.asarray(monk_test_list)
    test_labels = np.asarray(test_labels_list)

    return (train, train_labels, test, test_labels)
