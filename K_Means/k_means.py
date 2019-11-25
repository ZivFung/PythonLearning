import numpy as np
import matplotlib.pyplot as plt


def CalEuclideanDistance(Vector1, Vector2):
    return np.sqrt(np.sum(np.power(Vector1 - Vector2, 2)))


def InitCentroids(DataSet, k):
    SampleNum, Dim = DataSet.shape
    Centroid = np.zeros((k, Dim))
    for i in range(k):
        index = int(np.random.uniform(0, SampleNum))
        Centroid[i, :] = DataSet[index, :]
    return Centroid


def Kmeans(Dataset, k):
    Samplenum = Dataset.shape[0]

    ClusterAssment = np.mat(np.zeros((Samplenum, 2)))
    ClusterChanged = True

    Centroid = InitCentroids(Dataset, k)

    while ClusterChanged:
        ClusterChanged = False

        for i in range(Samplenum):
            minDist = 100000.0
            minIndex = 0

            for j in range(k):
                Distance = CalEuclideanDistance(Centroid[j, :], Dataset[i, :])
                if Distance < minDist:
                    minDist = Distance
                    minIndex = j

            if ClusterAssment[i, 0] != minIndex:
                ClusterChanged = True
                ClusterAssment[i, :] = minIndex, minDist**2

        for j in range(k):
            PointsInCluster = Dataset[np.nonzero(ClusterAssment[:, 0] == j)[0]]
            Centroid[j, :] = np.mean(PointsInCluster, axis=0)

    print('K-means Complete !')
    return Centroid, ClusterAssment


def ShowCluster(Dataset, k, Centroid, ClusterAssment):
    # plt.figure(figsize=(16, 8))
    plt.figure(1, figsize=(8, 6))
    plt.title("Dataset Before Clustering")
    plt.ylabel('Y')
    plt.xlabel('X')
    # plt.subplot(1, 2, 1), plt.title("Dataset Before Clustering"), plt.ylabel('Y'), plt.xlabel('X')
    for i in range(Dataset.shape[0]):
        plt.plot(Dataset[i, 0], Dataset[i, 1], 'or')

    SampleNum, Dim = Dataset.shape
    if Dim != 2:
        print("Dimension less than 2!")
        return 1
    
    Mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(Mark):
        print("The k is too large!")
        return 1
    
    # plt.subplot(1, 2, 2), plt.title("Dataset After Clustering"), plt.ylabel('Y'), plt.xlabel('X')
    plt.figure(2, figsize=(8, 6))
    plt.title("Dataset After Clustering")
    plt.ylabel('Y')
    plt.xlabel('X')
    for i in range(SampleNum):
        MarkIndex = np.int(ClusterAssment[i, 0])
        plt.plot(Dataset[i, 0], Dataset[i, 1], Mark[MarkIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']  
      
    for i in range(k):  
        plt.plot(Centroid[i, 0], Centroid[i, 1], mark[i], markersize=12) 

    plt.show()



