import numpy as np
import sys
import csv

trainingFile = "train.csv"
testingFile = "test_pub.csv"


def main():
    trainingDataFull = loadCsv(trainingFile)[:,1:]
    trainingData = trainingDataFull[:,:-1]
    
    testingData = loadCsv(testingFile)[:,1:]
    trainingDataFold = np.copy(trainingData)
    np.random.shuffle(trainingDataFold)
    print(np.shape(trainingData))
    fold1 = trainingDataFold[:1999]
    fold2 = trainingDataFold[2000:3999]
    fold3 = trainingDataFold[4000:5999]
    fold4 = trainingDataFold[6000:7999]
    
    print(testingData)
    kvals = [1,3,5,7,9,99,999,6000]
    for k in kvals:
        points = kNNAll(trainingData, trainingData, k, trainingDataFull)
        #print(points)
        accuracy = checkAccuracy(trainingDataFull, points)
        print("k =", k, accuracy)
    # k=1
    # # points = kNNAll(testingData, trainingData, k, trainingDataFull)
    # # writeCsv(points)
    # crossValidation(trainingDataFull, fold1, fold2, fold3, fold4, trainingData, k)
    # k=3
    # crossValidation(trainingDataFull, fold1, fold2, fold3, fold4, trainingData, k)
    # k=5
    # crossValidation(trainingDataFull, fold1, fold2, fold3, fold4, trainingData, k)
    # k=7
    # crossValidation(trainingDataFull, fold1, fold2, fold3, fold4, trainingData, k)
    # k=9
    # crossValidation(trainingDataFull, fold1, fold2, fold3, fold4, trainingData, k)
    # k=99
    # crossValidation(trainingDataFull, fold1, fold2, fold3, fold4, trainingData, k)
    # k=999
    # crossValidation(trainingDataFull, fold1, fold2, fold3, fold4, trainingData, k)
    # k=6000
    # crossValidation(trainingDataFull, fold1, fold2, fold3, fold4, trainingData, k)
    
    # for i in range(len(testingData)):
    #     neighbors = kNN(testingData[i], trainingData)
    #     classify(neighbors, i,trainingData,points)
    #print(points)



def loadCsv(file):
    return np.loadtxt(file, skiprows = 1, delimiter = ",")

#euclidean distance between 2 points
def eDist(newPoint, point):
    return np.linalg.norm(newPoint-point)

#k nearest neighbors for 1 new point
def kNN(z, x, k):
    neighbors = []
    for i in x:
        neighbors.append(eDist(z,i))
    indices = np.argsort(neighbors,axis = 0, kind = 'quicksort')
    return indices[:k]

def kNNAll(trainingData, fold, k, trainingDataFull):
    points = {}    
    for i in range(len(trainingData)):
        neighbors = kNN(trainingData[i], fold, k)
        classify(neighbors, i,trainingDataFull,points)
    return points
#
def classify(neighbors, point, trainingDataFull, pointsSet):
    sum = 0
    for j in neighbors:
        sum += trainingDataFull[j][-1]
    total = sum/len(neighbors)
    if(total >=.5):
        pointsSet[point] = 1
    else:
        pointsSet[point] = 0
    return pointsSet
        
#combines matrices
def combineFold(a, b, c):
    return np.concatenate((a,b,c))

def checkAccuracy(trainingData, points):
    sum = 0
    for i in range(len(points)):
        if trainingData[i][-1] == points[i]:
            sum +=1
    return sum/len(points)
    

def crossValidation(fullTrainingData, foldA, foldB, foldC, foldD, trainingData, k):
    subset1 = combineFold(foldB, foldC, foldD)
    subset2 = combineFold(foldA, foldC, foldD)
    subset3 = combineFold(foldA, foldB, foldD)
    subset4 = combineFold(foldA, foldB, foldC)
    
    pointSet1 = kNNAll(subset1, foldA, k, fullTrainingData)
    pointSet2 = kNNAll(subset2, foldB, k, fullTrainingData)
    pointSet3 = kNNAll(subset3, foldC, k, fullTrainingData)
    pointSet4 = kNNAll(subset4, foldD, k, fullTrainingData)
    
    accuracy1 = checkAccuracy(fullTrainingData, pointSet1)
    accuracy2 = checkAccuracy(fullTrainingData, pointSet2)
    accuracy3 = checkAccuracy(fullTrainingData, pointSet3)
    accuracy4 = checkAccuracy(fullTrainingData, pointSet4)
    
    var = np.var([accuracy1, accuracy2, accuracy3, accuracy4])
    ep = (accuracy1 + accuracy2 + accuracy3 + accuracy4)/4
    print("k =", k, accuracy1, accuracy2, accuracy3, accuracy4, "estimated validation performance: ", ep, "variance:", var)
    
 
def writeCsv(points):
    filename = "output.csv"
    fields = ['id', 'income']
    with open(filename, 'w+', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        for key, value in points.items():
            csvwriter.writerow([key, value])
        
    
if __name__ == "__main__":
    main()