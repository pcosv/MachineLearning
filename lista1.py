import csv
import math
import operator
import time
from decimal import Decimal
from numpy import array
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


# load data
def loadDataSet(file):
    with open(file) as csvfile:
        lines = list(csv.reader(csvfile, delimiter=','))
        return lines


# função que retorna a distância euclidiana
def getDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance = distance + pow((Decimal(instance1[x]) - Decimal(instance2[x])), 2)
    return math.sqrt(distance)


# retorna vizinhos de uma dada instância dado o k (kNN sem pesos)
def getNeighborsKnn(trainSet, instance, k):
    distances = []
    length = len(trainSet)-1
    for x in range(len(trainSet)):
        dist = getDistance(instance, trainSet[x], length)
        distances.append((trainSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getWeigth(distance):
    if distance == 0:
        return 0
    else:
        return 1/distance


# dado k, retorna vizinhos de uma dada instância com seus respectivos pesos (kNN com pesos)
def getNeighborsWeightedKnn(trainSet, instance, k):
    distances = []
    weights = []
    length = len(instance) - 1
    for x in range(len(trainSet)):
        dist = getDistance(instance, trainSet[x], length)
        weight = getWeigth(dist)
        weights.append(weight)
        distances.append((trainSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    weights.sort(reverse=True)

    distWeighted = []
    for x in range(len(distances)):
        distWeighted.append((distances[x][0], weights[x]))

    neighbors = []
    for x in range(k):
        neighbors.append(distWeighted[x])

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


# faz o mesmo que a função anterior, mas retorna a classe com maior somatorio de pesos
def getResponseWeightedKnn(neighbors):
    classes = {}
    sumTrue = 0
    sumFalse = 0
    for x in range(len(neighbors)):
        neighbor = neighbors[x][0]
        weigth = neighbors[x][1]
        response = neighbor[-1]
        if response in classes:
            if response == True:
                sumTrue = sumTrue + weigth
                classes[response] = sumTrue
            else:
                sumFalse = sumFalse + weigth
                classes[response] = sumFalse
        else:
            if response == True:
                classes[response] = weigth
                sumTrue = weigth
            else:
                classes[response] = weigth
                sumFalse = weigth
        classes.items()
        response = max(classes.items(), key=operator.itemgetter(1))[0]
    return response


def getAccuracy(testSet, results):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == results[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0


# plota graficos
def graph(k, result):
    auxPlot = []
    for y in k:
        auxPlot.append(result[y])
    plt.plot(k, auxPlot)
    plt.show()


# função principal
def main():

    # Escolha qual algoritmo deseja executar
    # 0 - kNN
    # 1 - Distance-Weighted kNN
    algorithm = 1

    # dicionario que ao final da execução dara as taxas de acerto para cada valor de k
    result = {}

    # carregar dados do csv dado como entrada
    # caso deseje executar o outro BD, substitua a string passada como parâmetro por 'kc1.csv'
    data = array(loadDataSet('cm1.csv'))

    # 10 folds
    kfold = KFold(10, False, 1)

    # valores de k
    k = [1, 2, 3, 5, 7, 9, 11, 13, 15]

    accuracyArray = []

    for i in k:
        for train, test in kfold.split(data):
            responses = []
            for x in range(len(data[test])):
                # chama funções dependendo do algoritmo escolhido
                if algorithm == 0:
                    neighbors = getNeighborsKnn(data[train], data[test][x], i)
                    response = getResponseKnn(neighbors)
                elif algorithm == 1:
                    neighbors = getNeighborsWeightedKnn(data[train], data[test][x], i)
                    response = getResponseWeightedKnn(neighbors)
                responses.append(response)

            accuracy = getAccuracy(data[test], responses)
            accuracyArray.append(accuracy)
            responses.clear()
        result[i] = sum(accuracyArray) / len(accuracyArray)
    # exibe no console dicionario com resultados, sendo k a chave e a taxa de acerto como valor
    print(result)

    # desenha grafico com a variação da taxa de acerto em relação ao k
    graph(k, result)

start = time.time()
main()
end = time.time()
# tempo de processamento
print(end - start)