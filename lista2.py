import csv
import operator
import time
from math import sqrt
from random import randrange
import random
from numpy import array
import copy
import matplotlib.pyplot as plt



# load data
def loadDataSet(file):
    with open(file) as csvfile:
        lines = list(csv.reader(csvfile, delimiter=','))
        return lines

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup



# distancia euclidiana entre dois vetores
def getDistance(instance1, instance2):
    distance = 0
    for i in range(len(instance1) - 1):
        distance += ((instance1[i]) - (instance2[i])) ** 2
    return sqrt(distance)


# LVQ1

# encontra o prototipo "vizinho" de cada elemento do conjunto original
def getNeighborPrototype(prototypes, test_row):
    distances = list()
    for prototype in prototypes:
        dist = getDistance(prototype, test_row)
        distances.append((prototype, dist))
    distances.sort(key=lambda tup: tup[1])
    return distances[0][0]

# Create a random prototype vector
def random_prototype(train):
    n_records = len(train)
    n_features = len(train[0])
    prototype = [train[randrange(n_records)][i] for i in range(n_features)]
    return prototype

# esta função recebe os dados, o numero de protótipos, a taxa de aprendizado e a quantidade de épocas e
# retorna o conjunto de protótipos final
def trainPrototypesLVQ1(train, n_prototypes, lrate, epochs):
    prototypes = [random_prototype(train) for i in range(n_prototypes)]
    for epoch in range(epochs):
        rate = lrate * (1.0 - (epoch / float(epochs)))
        for row in train:
            neighborPrototype = getNeighborPrototype(prototypes, row)
            for i in range(len(row) - 1):
                error = (row[i]) - (neighborPrototype[i])
                ## se as classes são iguais, aproxima, do contrario afasta
                if neighborPrototype[-1] == row[-1]:
                    neighborPrototype[i] += rate * error
                else:
                    neighborPrototype[i] -= rate * error
        #print('>epoch=%d, lrate=%.3f' % (epoch, rate))
    return prototypes

# LVQ2

# retorna os dois vizinhos mais proximos
def getNeighborsPrototypes(prototypes, test_row):
    distances = list()
    finalArrayDist = []
    for prototype in prototypes:
        dist = getDistance(prototype, test_row)
        distances.append((prototype, dist))
    distances.sort(key=lambda tup: tup[1])
    finalArrayDist.append(distances[1])
    finalArrayDist.append(distances[2])
    return finalArrayDist

# define se o prototipo esta dentro da janela
def s(w):
    s = (1.0 - w)/(1.0 + w)
    return s


# define se o prototipo esta na janela aceitavel
def window(neighbors, w):
    di = neighbors[0][1]
    dj = neighbors[1][1]
    a = di/dj
    b = dj/di
    minimum = min(a,b)
    #print(minimum)
    if minimum > s(w):
        return True
    else:
        return False


def trainPrototypesLVQ2_1(train, n_prototypes, lrate, epochs, w):
    prototypes = copy.deepcopy(train)
    for epoch in range(epochs):
        rate = lrate * (1.0 - (epoch / float(epochs)))
        for row in train:
            neighborsPrototypes = getNeighborsPrototypes(prototypes, row)
            n1 = neighborsPrototypes[0][0]
            n2 = neighborsPrototypes[1][0]
            isWindow = window(neighborsPrototypes, w)
            if isWindow == False or n1[-1] == n2[-1]:
                continue
            else:
                for i in range(len(row) - 2):
                    error = (row[i]) - (n1[i])
                    # se as classes são iguais, aproxima, do contrario afasta
                    if n1[-1] == row[-1]:
                        n1[i] += rate * error
                    else:
                        n1[i] -= rate * error

        #print('>epoch=%d, lrate=%.3f' % (epoch, rate))
    return prototypes


# LVQ3

# o LVQ é identico ao LVQ2.1 mas aproxima (a um fator e) ambos os vizinhos caso eles sejam da mesma classe do prototipo
def trainPrototypesLVQ3(train, n_prototypes, lrate, epochs, w, e):
    prototypes = copy.deepcopy(train)
    for epoch in range(epochs):
        rate = lrate * (1.0 - (epoch / float(epochs)))
        for row in train:
            neighborsPrototypes = getNeighborsPrototypes(prototypes, row)
            n1 = neighborsPrototypes[0][0]
            n2 = neighborsPrototypes[1][0]
            isWindow = window(neighborsPrototypes, w)
            neighbors = [n1, n2]
            if isWindow == False or (n1[-1] == n2[-1] and (n1[-1] != row[-1])):
                continue
            elif n1[-1] == n2[-1] == row[-1]:
                for neighbor in neighbors:
                    for i in range(len(row) - 2):
                        error = (row[i]) - (neighbor[i])
                        neighbor[i] += e * rate * error
            else:
                for neighbor in neighbors:
                    for i in range(len(row) - 2):
                        error = (row[i]) - (neighbor[i])
                        # se as classes são iguais, aproxima, do contrario afasta
                        if neighbor[-1] == row[-1]:
                            neighbor[i] += rate * error
                        else:
                            neighbor[i] -= rate * error
        #print('>epoch=%d, lrate=%.3f' % (epoch, rate))
    return prototypes

# k-NN implementado abaixo

# retorna vizinhos de uma dada instância dado o k
def getNeighborsKnn(trainSet, instance, k):
    distances = []
    for x in range(len(trainSet)):
        dist = getDistance(instance, trainSet[x])
        distances.append((trainSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

# dados os vizinhos, esta função calcula a classe predita pelo algoritmo (kNN sem pesos)
def getResponseKnn(neighbors):
    classes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classes:
            classes[response] += 1
        else:
            classes[response] = 1
        classes.items()
        # pega o valor maximo do dicionario, ou seja, a classe com mais aparições entre os vizinhos da instância de teste
        response = max(classes.items(), key=operator.itemgetter(1))[0]
    return response

# calcula a acuracia do k-NN
def getAccuracy(testSet, results):
    correct = 0
    for x in range(len(testSet) - 1):
        if testSet[x][-1] == results[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0

# função responsável por plotar os resultados no gráfico
def draw(algorithms, results, k, n_prototypes, dataset):
    plt.plot(algorithms, results, 'ro')
    title = "K = " + str(k) + " e n = " + str(n_prototypes) + " na base de dados " + str(dataset)
    plt.suptitle(title, fontsize=13, fontweight=0, color='black',
                 style='italic', y=0.95)
    plt.ylabel('Acurácia')
    plt.xlabel('Algoritmo de Seleção')
    plt.show()


# função que carrega e prepara os dados
def prepareData():
    data = loadDataSet('cm1.csv')
    for i in range(len(data[0]) - 1):
        str_column_to_float(data, i)
    # convert class column to integers
    str_column_to_int(data, len(data[0]) - 1)
    return data


# função principal
def main():
    # setando a base de dados
    dataSetName = 'cm1.csv'
    data = prepareData()
    # separate data
    random.shuffle(data)
    data = array(data)
    split = int(3*(len(data))/4)
    #print(split)
    train, test = data[:split, :], data[split:, :]

    # All algorithms
    learn_rate = 0.3
    n_epochs = 10
    # LVQ2.1
    w = 0.6
    # LVQ3
    e = 0.5

    algorithms = ['Sem','LVQ1', 'LVQ2.1', 'LVQ3']
    n_prototypes = [10, 50, 100, 200]
    dictFinalResult = {}
    k = 3

    for n in n_prototypes:
        prototypesLVQ1 = trainPrototypesLVQ1(train, n, learn_rate, n_epochs)
        prototypesLVQ2_1 = trainPrototypesLVQ2_1(prototypesLVQ1, n, learn_rate, n_epochs, w)
        prototypesLVQ3 = trainPrototypesLVQ3(prototypesLVQ1, n, learn_rate, n_epochs, w, e)
        results = []
        datasets = [train, prototypesLVQ1, prototypesLVQ2_1, prototypesLVQ3]
        for set in datasets:
            responses = []
            for x in range(len(test) - 1):
                neighbors = getNeighborsKnn(set, test[x], k)
                response = getResponseKnn(neighbors)
                responses.append(response)
            accuracy = getAccuracy(test, responses)
            responses.clear()
            results.append(accuracy)
        dictFinalResult[n] = results

        #draw(algorithms, results, k, n, dataSetName)

    # Ao final da execução, dicionario com resultados sera mostrado (a chave é o numero de prototipos o valor é
    # um array com as acuracias para os dados sem tratamento, com LVQ1, LVQ2.1 e LVQ3 respectivamente
    print('Resultados')
    print(dictFinalResult)

main()


